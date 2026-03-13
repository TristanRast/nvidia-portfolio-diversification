#!/usr/bin/env python3
"""
Master Script: Run Complete NVIDIA Portfolio Diversification Analysis

This script executes the entire analysis pipeline from data collection to visualization.

Author: Tristan Rast
Date: March 11, 2026
WGU Capstone Project - BHN1 Task 3

Usage:
    python run_analysis.py              # Run full pipeline
    python run_analysis.py --skip-data  # Skip data collection (use existing)
    python run_analysis.py --viz-only   # Generate visualizations only
"""

import sys
import os
import time
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_script(script_name, script_number):
    """
    Run a single analysis script.

    Parameters
    ----------
    script_name : str
        Name of the script to run
    script_number : int
        Script number in sequence
    """
    logger.info("")
    logger.info("="*80)
    logger.info(f"STEP {script_number}: {script_name}")
    logger.info("="*80)

    script_path = os.path.join('src', script_name)

    try:
        start_time = time.time()

        # Execute the script
        with open(script_path, 'r') as f:
            code = f.read()

        exec(compile(code, script_path, 'exec'), {'__name__': '__main__', '__file__': str(script_path)})

        elapsed = time.time() - start_time
        logger.info(f"✓ {script_name} completed successfully in {elapsed:.2f} seconds")

        return True

    except Exception as e:
        logger.error(f"[FAILED] Error in {script_name}: {str(e)}")
        logger.error(f"Pipeline halted at Step {script_number}")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run NVIDIA Portfolio Diversification Analysis Pipeline'
    )
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data collection step (use existing data)')
    parser.add_argument('--viz-only', action='store_true',
                       help='Run visualization only (assumes data already processed)')

    args = parser.parse_args()

    # Define pipeline steps
    all_steps = [
        ('01_data_collection.py', 'Data Collection'),
        ('02_data_preparation.py', 'Data Preparation'),
        ('03_correlation_analysis.py', 'Correlation Analysis'),
        ('04_rolling_correlation.py', 'Rolling Correlation Analysis'),
        ('05_portfolio_analysis.py', 'Portfolio Analysis'),
        ('06_visualization.py', 'Visualization Generation'),
    ]

    # Determine which steps to run
    if args.viz_only:
        steps = [all_steps[-1]]  # Only visualization
        logger.info("Running visualization only (using existing data)")
    elif args.skip_data:
        steps = all_steps[1:]  # Skip data collection
        logger.info("Skipping data collection (using existing raw data)")
    else:
        steps = all_steps  # Full pipeline
        logger.info("Running complete analysis pipeline")

    # Print banner
    logger.info("")
    logger.info("="*80)
    logger.info("NVIDIA PORTFOLIO DIVERSIFICATION ANALYSIS")
    logger.info("WGU Data Analytics Capstone - BHN1 Task 3")
    logger.info("Author: Tristan Rast")
    logger.info("="*80)
    logger.info(f"Total steps to execute: {len(steps)}")
    logger.info("")

    # Execute pipeline
    overall_start = time.time()

    for i, (script_file, description) in enumerate(steps, start=1):
        success = run_script(script_file, i)
        if not success:
            logger.error("\n[PIPELINE FAILED]")
            logger.error(f"Failed at: {description}")
            logger.error("Please check the error messages above and fix before re-running.")
            return False

        # Small pause between steps
        time.sleep(1)

    # Pipeline complete
    overall_elapsed = time.time() - overall_start
    minutes = int(overall_elapsed // 60)
    seconds = int(overall_elapsed % 60)

    logger.info("")
    logger.info("="*80)
    logger.info("[OK] PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Total execution time: {minutes}m {seconds}s")
    logger.info("")
    logger.info("Output Locations:")
    logger.info("   - Processed data: data/processed/")
    logger.info("   - Analysis results: data/results/")
    logger.info("   - Visualizations: outputs/figures/")
    logger.info("   - Reports: outputs/reports/")
    logger.info("")
    logger.info("Next Steps:")
    logger.info("   1. Review visualizations in outputs/figures/")
    logger.info("   2. Review analysis results in data/results/")
    logger.info("   3. Use findings to write your final report")
    logger.info("   4. Record Panopto presentation demonstrating the code")
    logger.info("")

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
