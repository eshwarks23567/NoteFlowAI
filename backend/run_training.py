"""Run all 4 ML training pipelines sequentially."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("  RUNNING ALL TRAINING PIPELINES")
    print("=" * 60)

    # Pipeline 1: Slide OCR
    print("\n>>> Pipeline 1/4: Slide OCR")
    from training.train_slide_ocr import OCRTrainingPipeline, OCRTrainingConfig
    ocr = OCRTrainingPipeline(OCRTrainingConfig(epochs=10, batch_size=16))
    ocr_result = ocr.train()

    # Pipeline 2: Gesture Recognition
    print("\n>>> Pipeline 2/4: Gesture Recognition")
    from training.train_gesture import GestureTrainingPipeline, GestureTrainingConfig
    gesture = GestureTrainingPipeline(GestureTrainingConfig(epochs=15))
    gesture_result = gesture.train()

    # Pipeline 3: Voice Emphasis
    print("\n>>> Pipeline 3/4: Voice Emphasis Detection")
    from training.train_emphasis import EmphasisTrainingPipeline, EmphasisTrainingConfig
    emphasis = EmphasisTrainingPipeline(EmphasisTrainingConfig(epochs=15))
    emphasis_result = emphasis.train()

    # Pipeline 4: Multimodal Fusion
    print("\n>>> Pipeline 4/4: Multimodal Fusion")
    from training.train_fusion import FusionTrainingPipeline, FusionTrainingConfig
    fusion = FusionTrainingPipeline(FusionTrainingConfig(epochs=20))
    fusion_result = fusion.train()

    # Summary
    print("\n" + "=" * 60)
    print("  ALL 4 PIPELINES COMPLETE")
    print("=" * 60)
    print(f"  OCR model:      {ocr_result['model_path']}")
    print(f"  Gesture model:  {gesture_result['model_path']}")
    print(f"  Emphasis model: {emphasis_result['model_path']}")
    print(f"  Fusion model:   {fusion_result['model_path']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
