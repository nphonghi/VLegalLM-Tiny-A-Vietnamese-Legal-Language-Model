#!/usr/bin/env python3
"""
VLexRIA Inference Module

Provides inference capabilities including:
1. Text generation (greedy, sampling, beam search)
2. MTP (Multi-Token Prediction) speculative decoding
3. Batch inference
4. Interactive chat mode

Usage:
    python inference.py --checkpoint artifacts/checkpoints/sft/legal_vn/best.pt --prompt "Xin chào!"
    python inference.py --checkpoint artifacts/checkpoints/sft/legal_vn/best.pt --interactive
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F

from vlex_ria.core.config import load_config, InferenceConfig, get_device
from vlex_ria.model import VLexRIAModel
from vlex_ria.data import get_tokenizer
from vlex_ria.core.utils import get_logger

# Initialize logger
logger = get_logger(__name__)

class VLexRIAInference:
    """
    Inference wrapper for VLexRIA model.
    
    Supports:
    - Standard autoregressive generation
    - MTP speculative decoding for faster inference
    - Various sampling strategies
    """
    
    def __init__(
        self,
        model: VLexRIAModel,
        tokenizer: Any,
        config: InferenceConfig,
        device: str = "auto",
    ):
        """
        Args:
            model: Trained VLexRIA model
            tokenizer: Tokenizer
            config: Inference configuration
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.device = torch.device(get_device(device))
        self.model.to(self.device)
        self.model.eval()
        
        # Get special token ids
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        use_mtp: Optional[bool] = None,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-K sampling
            do_sample: Whether to sample (vs greedy)
            repetition_penalty: Penalty for repeating tokens
            use_mtp: Use MTP speculative decoding
            stream: Stream tokens as they are generated
            
        Returns:
            Generated text
        """
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        repetition_penalty = repetition_penalty or self.config.repetition_penalty
        use_mtp = use_mtp if use_mtp is not None else self.config.use_mtp_decoding
        stream = stream if stream is not None else self.config.stream
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        if use_mtp and self.model.mtp_enabled:
            if stream:
                output_gen = self._generate_with_mtp(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    stream=True,
                )
                # Collect streamed tokens
                generated_text = ""
                for token in output_gen:
                    if token is not None:
                        generated_text += token
                return generated_text
            else:
                output_ids = self._generate_with_mtp(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    stream=False,
                )
                # Decode
                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                return generated_text
        else:
            if stream:
                output_gen = self._generate_standard(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    stream=True,
                )
                # Collect streamed tokens
                generated_text = ""
                for token in output_gen:
                    if token is not None:
                        generated_text += token
                return generated_text
            else:
                output_ids = self._generate_standard(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    stream=False,
                )
                # Decode
                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                return generated_text
    
    def _generate_standard(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: float,
        stream: bool = False,
    ) -> torch.Tensor:
        """Standard autoregressive generation."""
        generated = input_ids
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated
            
            outputs = self.model(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs['logits'][:, -1, :]  # (B, V)
            past_key_values = outputs['past_key_values']
            
            # Apply repetition penalty (vectorized)
            if repetition_penalty != 1.0:
                for b in range(generated.shape[0]):
                    unique_tokens = generated[b].unique()
                    logits[b, unique_tokens] /= repetition_penalty
            
            # Sample next token
            next_token = self._sample_token(logits, temperature, top_p, top_k, do_sample)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if next_token[0, 0].item() == self.eos_token_id:
                if stream:
                    yield None  # Signal end of generation
                break
            
            # Stream token if requested
            if stream:
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
                if token_text:
                    yield token_text
        
        if not stream:
            return generated
    
    def _generate_with_mtp(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: float,
        stream: bool = False,
    ) -> torch.Tensor:
        """
        Generation with MTP speculative decoding.
        
        Algorithm (speculate-then-verify):
        1. Run model to get main logits + MTP speculative logits
        2. Sample main token + N speculative tokens from MTP heads
        3. Verify speculative tokens with a single forward pass
        4. Accept the longest prefix of matching tokens
        5. Cache verify logits and truncate KV to last accepted position
        
        This generates up to (1 + N) tokens per verification call when
        MTP heads agree, falling back to 1 token/call when they disagree.
        
        Key invariant: after each iteration, past_key_values contains KV
        entries for ALL tokens in ``generated``. The cached verify logits
        provide the next-token prediction without a redundant forward pass,
        preventing KV cache duplication.
        """
        generated = input_ids
        past_key_values = None
        num_generated = 0
        
        num_mtp_heads = self.model.config.mtp.num_predict_tokens
        
        # Cached logits from the previous verify pass.
        # After verification, verify_logits[:, num_accepted-1, :] gives the
        # model's prediction for the NEXT token without needing another
        # forward pass. This avoids the KV cache duplication bug: accepted
        # tokens are already in the KV cache, so feeding generated[:, -1:]
        # again would create duplicate entries and corrupt generation.
        _cached_main_logits = None   # (1, V) — reusable next-token logits
        _cached_mtp_logits = None    # List[(1, V)] — reusable MTP logits
        
        while num_generated < max_new_tokens:
            # Step 1: Obtain main + MTP logits
            if _cached_main_logits is not None:
                # Reuse logits from the previous verify pass (no forward
                # pass needed — the KV cache already covers all of
                # ``generated``, and these logits predict the next token).
                main_logits = _cached_main_logits
                mtp_logits_extracted = _cached_mtp_logits   # List[(1, V)]
                _cached_main_logits = None
                _cached_mtp_logits = None
            else:
                # Fresh forward pass (first iteration or non-MTP fallback)
                if past_key_values is not None:
                    curr_input = generated[:, -1:]
                else:
                    curr_input = generated
                
                outputs = self.model(
                    input_ids=curr_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                main_logits = outputs['logits'][:, -1, :]   # (1, V)
                raw_mtp = outputs.get('mtp_logits', None)
                past_key_values = outputs['past_key_values']
                
                # Pre-extract MTP logits at last position → (1, V) each
                if raw_mtp:
                    mtp_logits_extracted = [
                        pred[:, -1, :] for pred in raw_mtp
                    ]
                else:
                    mtp_logits_extracted = None
            
            # Apply repetition penalty — clone first to avoid mutating
            # cached tensors that may be reused.
            if repetition_penalty != 1.0:
                main_logits = main_logits.clone()
                unique_tokens = generated[0].unique()
                main_logits[0, unique_tokens] /= repetition_penalty
            
            # Sample main token (always accepted)
            main_token = self._sample_token(
                main_logits, temperature, top_p, top_k, do_sample
            )
            
            # If no MTP predictions available, fall back to standard decode
            if mtp_logits_extracted is None:
                generated = torch.cat([generated, main_token], dim=1)
                num_generated += 1
                
                if stream:
                    token_text = self.tokenizer.decode(
                        main_token[0], skip_special_tokens=False
                    )
                    if token_text:
                        yield token_text
                
                if main_token[0, 0].item() == self.eos_token_id:
                    if stream:
                        yield None
                    if not stream:
                        return generated
                    return
                continue
            
            # Collect draft tokens: [main_token, spec_1, ..., spec_N]
            draft_tokens = [main_token]
            for mtp_logit in mtp_logits_extracted:
                spec_token = self._sample_token(
                    mtp_logit, temperature, top_p, top_k, do_sample
                )
                draft_tokens.append(spec_token)
            
            # Step 2: Verify speculative tokens
            # Feed all draft tokens through the model in a single pass.
            # verify_logits[:, i, :] is the model's autoregressive
            # prediction for the token AFTER draft_tokens[i].
            draft_ids = torch.cat(draft_tokens, dim=1)   # (1, 1+N)
            
            verify_outputs = self.model(
                input_ids=draft_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            verify_logits = verify_outputs['logits']      # (1, 1+N, V)
            verify_mtp = verify_outputs.get('mtp_logits', None)
            verify_kv = verify_outputs['past_key_values']
            
            # Compare each speculative token with the model's prediction.
            # Use the SAME sampling strategy as drafting for consistency;
            # greedy verify with sampled draft causes unnecessary rejections.
            num_accepted = 1   # main_token is always accepted
            
            for i in range(len(draft_tokens) - 1):
                verify_token = self._sample_token(
                    verify_logits[:, i, :],
                    temperature, top_p, top_k, do_sample,
                )
                
                if verify_token[0, 0].item() == draft_tokens[i + 1][0, 0].item():
                    num_accepted += 1
                else:
                    break
            
            # Step 3: Accept tokens and update generated
            accepted_ids = draft_ids[:, :num_accepted]     # (1, num_accepted)
            generated = torch.cat([generated, accepted_ids], dim=1)
            num_generated += num_accepted
            
            # Stream accepted tokens
            if stream:
                for i in range(num_accepted):
                    token_text = self.tokenizer.decode(
                        draft_tokens[i][0], skip_special_tokens=False
                    )
                    if token_text:
                        yield token_text
            
            # Check for EOS among accepted tokens
            eos_found = False
            for i in range(num_accepted):
                if draft_tokens[i][0, 0].item() == self.eos_token_id:
                    eos_found = True
                    break
            
            if eos_found:
                if stream:
                    yield None
                if not stream:
                    return generated
                return
            
            # Step 4: Truncate KV cache to accepted length
            # verify_kv contains entries for:
            #   [previous_context] + [all 1+N draft tokens]
            # We keep only:
            #   [previous_context] + [num_accepted tokens]
            num_draft = len(draft_tokens)
            num_to_remove = num_draft - num_accepted
            
            if num_to_remove > 0:
                truncated_kv = []
                for layer_kv in verify_kv:
                    if isinstance(layer_kv, tuple) and len(layer_kv) == 2:
                        k, v = layer_kv
                        truncated_kv.append((
                            k[:, :, :-num_to_remove, :],
                            v[:, :, :-num_to_remove, :],
                        ))
                    else:
                        truncated_kv.append(layer_kv)
                past_key_values = truncated_kv
            else:
                past_key_values = verify_kv
            
            # Step 5: Cache verify logits for next iteration
            # CRITICAL: past_key_values now includes ALL accepted tokens.
            # If the next iteration ran a fresh forward pass with
            # generated[:, -1:], the last accepted token would be
            # processed AGAIN, creating a duplicate KV entry and
            # corrupting all subsequent generation.
            #
            # Fix: verify_logits[:, num_accepted-1, :] is the model's
            # prediction for the token AFTER the last accepted token.
            # We cache this so the next iteration can skip the forward
            # pass entirely, keeping the KV cache in sync.
            _cached_main_logits = verify_logits[:, num_accepted - 1, :]
            
            if verify_mtp:
                _cached_mtp_logits = [
                    pred[:, num_accepted - 1, :]
                    for pred in verify_mtp
                ]
            else:
                _cached_mtp_logits = None
            
            if num_generated >= max_new_tokens:
                if stream:
                    yield None
                break
        
        if not stream:
            return generated
    
    def _sample_token(
        self,
        logits: torch.Tensor,  # (B, V)
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> torch.Tensor:
        """Sample a single token from logits."""
        # Temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        
        if not do_sample:
            return torch.argmax(logits, dim=-1, keepdim=True)
        
        # Top-K filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    def chat(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Chat-style generation with conversation history.
        
        Args:
            user_message: User's input message
            history: List of {"role": "user/assistant", "content": "..."} dicts
            system_prompt: Optional system prompt
            **kwargs: Generation parameters
            
        Returns:
            Assistant's response
        """
        # Build conversation prompt
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n\n")
        
        if history:
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    prompt_parts.append(f"Human: {content}\n\n")
                else:
                    prompt_parts.append(f"Assistant: {content}\n\n")
        
        prompt_parts.append(f"Human: {user_message}\n\nAssistant:")
        
        full_prompt = "".join(prompt_parts)
        
        # Generate
        response = self.generate(full_prompt, **kwargs)
        
        # Extract assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    
    def interactive_chat(self, system_prompt: Optional[str] = None):
        """Run interactive chat session with streaming output."""
        print("\n" + "=" * 100)
        print("VLexRIA Interactive Chat")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'clear' to reset conversation history")
        print("=" * 100 + "\n")
        
        history = []
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                history = []
                print("Conversation history cleared.\n")
                continue
            
            # Build conversation prompt
            prompt_parts = []
            
            if system_prompt:
                prompt_parts.append(f"System: {system_prompt}\n\n")
            
            if history:
                for turn in history:
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    if role == "user":
                        prompt_parts.append(f"Human: {content}\n\n")
                    else:
                        prompt_parts.append(f"Assistant: {content}\n\n")
            
            prompt_parts.append(f"Human: {user_input}\n\nAssistant:")
            
            full_prompt = "".join(prompt_parts)
            
            # Stream generation output
            print("Assistant: ", end="", flush=True)
            
            # Tokenize prompt
            input_ids = self.tokenizer.encode(full_prompt, return_tensors='pt').to(self.device)
            
            # Stream generate
            response_tokens = []
            generated = input_ids
            past_key_values = None
            
            max_new_tokens = self.config.max_new_tokens
            temperature = self.config.temperature
            top_p = self.config.top_p
            top_k = self.config.top_k
            do_sample = self.config.do_sample
            repetition_penalty = self.config.repetition_penalty
            
            for step in range(max_new_tokens):
                # Forward pass
                if past_key_values is not None:
                    curr_input = generated[:, -1:]
                else:
                    curr_input = generated
                
                outputs = self.model(
                    input_ids=curr_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                logits = outputs['logits'][:, -1, :]  # (B, V)
                past_key_values = outputs['past_key_values']
                
                # Apply repetition penalty (vectorized)
                if repetition_penalty != 1.0:
                    for b in range(generated.shape[0]):
                        unique_tokens = generated[b].unique()
                        logits[b, unique_tokens] /= repetition_penalty
                
                # Sample next token
                next_token = self._sample_token(logits, temperature, top_p, top_k, do_sample)
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
                response_tokens.append(next_token[0, 0].item())
                
                # Check for EOS
                if next_token[0, 0].item() == self.eos_token_id:
                    break
                
                # Decode and print the latest token
                # To properly handle multi-token words, we decode all response tokens so far
                full_response = self.tokenizer.decode(torch.tensor(response_tokens), skip_special_tokens=True)
                
                # Extract assistant's response
                if "Assistant:" in full_response:
                    assistant_response = full_response.split("Assistant:")[-1]
                else:
                    assistant_response = full_response
                
                # Print only the new part since last iteration
                if not hasattr(self, '_last_printed_response'):
                    self._last_printed_response = ""
                
                new_text = assistant_response[len(self._last_printed_response):]
                if new_text:
                    print(new_text, end="", flush=True)
                
                self._last_printed_response = assistant_response
            
            print("\n")
            
            # Reset last printed response
            self._last_printed_response = ""
            
            # Get final response for history
            full_response = self.tokenizer.decode(torch.tensor(response_tokens), skip_special_tokens=True)
            
            # Extract assistant's response
            if "Assistant:" in full_response:
                response = full_response.split("Assistant:")[-1].strip()
            else:
                response = full_response
            
            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})


def load_model_for_inference(
    checkpoint_path: str,
    config_path: Optional[str] = None,
) -> tuple:
    """
    Load model and tokenizer for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file
        
    Returns:
        model, tokenizer, config
    """
    # Load config
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config_path = Path(checkpoint_path).parent.parent / "configs" / "config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            from vlex_ria.core.config import VLexRIAConfig
            config = VLexRIAConfig()
    
    # Load tokenizer
    tokenizer = get_tokenizer(config.data)
    config.model.vocab_size = len(tokenizer)
    
    # Create model
    model = VLexRIAModel(config.model)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("Using randomly initialized model")
    
    return model, tokenizer, config


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="VLexRIA Inference")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/pretrain/legal_vn/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info("Loading model...")
    model, tokenizer, config = load_model_for_inference(
        args.checkpoint,
        args.config,
    )
    
    # Update inference config
    config.inference.max_new_tokens = args.max_new_tokens
    config.inference.temperature = args.temperature
    config.inference.top_p = args.top_p
    config.inference.device = args.device
    
    # Create inference wrapper
    inference = VLexRIAInference(
        model=model,
        tokenizer=tokenizer,
        config=config.inference,
        device=args.device,
    )
    
    # Run inference
    if args.interactive:
        inference.interactive_chat()
    elif args.prompt:
        logger.info(f"Prompt: {args.prompt}")
        logger.info("-" * 50)
        response = inference.generate(args.prompt)
        logger.info(f"Generated:\n{response}")
    else:
        # Demo generation
        prompts = [
            "Hỏi: Người lao động đơn phương chấm dứt hợp đồng lao động trái pháp luật thì có được nhận trợ cấp thôi việc không?",
            "Theo Bộ luật Hình sự 2015, tội cố ý gây thương tích",
            "Trách nhiệm bồi thường thiệt hại ngoài hợp đồng",
            "Mức phạt vi phạm nồng độ cồn khi lái xe",
        ]
        
        logger.info("Demo Generation:")
        logger.info("=" * 100)
        
        for prompt in prompts:
            logger.info(f"Prompt: {prompt}")
            logger.info("-" * 50)
            response = inference.generate(prompt, max_new_tokens=100)
            logger.info(f"Generated:\n{response}")


if __name__ == "__main__":
    main()
