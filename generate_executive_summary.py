\
from src.dashboard import build_executive_summary, plot_model_comparison

if __name__ == "__main__":
    summary = build_executive_summary()
    print(f"Executive summary saved to: {summary}")
    try:
        plot = plot_model_comparison()
        print(f"Model comparison plot saved to: {plot}")
    except FileNotFoundError:
        print("Metrics file not found. Run the analysis pipeline first.")
