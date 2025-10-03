Code_Pirates_Hackathon_PS-4
# Airdopes Assembly Verifier — Hackathon Demo

## Quick start (simulation mode - no API needed)
1. Install: `pip install -r requirements.txt`
2. Put videos in `data/` (correct.mp4, wrong1.mp4, wrong2.mp4).
3. Extract frames & run verification:
   `python src/pipeline.py --video data/correct.mp4 --outdir data/frames_correct --stride 8`
4. Start dashboard:
   `streamlit run src/app_streamlit.py`
   Upload `data/frames_correct/verification_result.json`.

## To use a real VLLM API
- Open `src/vllm_reasoner.py` and replace `vllm_query_api_template` with your actual client code (OpenAI / HuggingFace).
- Then run pipeline with `--use_api`.

## Notes
- The pipeline uses a simulated VLLM by default so you can demo immediately.
- Optionally add YOLOv8 detection for higher precision (see comments).
