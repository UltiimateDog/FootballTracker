#!/usr/bin/env python3
"""
Test script for player analysis functionality
"""

from Final.player_analysis import analyze_players_from_video
from pathlib import Path

def test_player_analysis():
    """Test player analysis with crop extraction and speed calculation"""
    
    # Paths
    input_path = "test_content/demo2.mp4"
    model_path = "models/ball_and_player_model.pt"
    output_path = "test_results/player_analysis_summary_3.jpg"
    
    # Check if files exist
    if not Path(input_path).exists():
        print(f"❌ Input video not found: {input_path}")
        return
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("🔍 Starting player analysis...")
    print(f"Input: {input_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print("\n📊 Features:")
    print("• Individual player identification")
    print("• Best quality crop extraction per player")
    print("• Average speed calculation (km/h)")
    print("• Team assignment")
    print("• Detection statistics")
    
    try:
        # Analyze video and create summary image
        analyze_players_from_video(
            input_path=input_path,
            model_path=model_path,
            output_path=output_path
        )
        
        print("\n✅ Player analysis completed!")
        print(f"Check the summary image: {output_path}")
        print("\n🎯 Summary includes:")
        print("• Player crop images with team color borders")
        print("• Player ID numbers")
        print("• Team assignments (Team A/B)")
        print("• Average speed in km/h")
        print("• Total detection count per player")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    test_player_analysis()