#!/bin/bash

# GPT-4o Mini Evaluation - All 16 Domains
# Author: Kazi
# Date: Nov 2024

echo "================================================================================"
echo "                   GPT-4o Mini Evaluation - All 16 Domains"
echo "================================================================================"
echo ""
echo "This will:"
echo "  • Generate 10 tasks per domain (160 total)"
echo "  • Run GPT-4o Mini with AND without context"
echo "  • Total runtime: ~1.5-2 hours"
echo "  • Total cost: ~\$0.15-0.25"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5
echo ""

# All 16 domains (matching your existing results structure)
domains=(
    "CashDepletedinATMScenario"
    "ConstrainedRandomWalk"
    "DecreaseInTraffic"
    "DirectNormalIrradianceFromCloudStatus"
    "ElectricityIncreaseInPredictionTask"
    "FullCausalContextExplicitEquationBivarLinSVAR"
    "ImplicitTrafficForecastTaskwithHolidays"
    "LaneClosureAfterShortHistoryMediumBackground"
    "OracleePredUnivariateConstraints"
    "PredictableSpikes"
    "PreedictableGrocerPersistentShockUnivariate"
    "STLPredTrendMultiplierWithMediumDescription"
    "SensorMaintenance"
    "SolarPowerProduction"
    "SpeedFromLoadTask"
    "UnemploymentCountyUsingSingleStateData"
)

total_domains=${#domains[@]}
current=0

start_time=$(date +%s)

# Run with context for all domains
echo "================================================================================"
echo "PHASE 1: Running WITH CONTEXT (${total_domains} domains)"
echo "================================================================================"
echo ""

for domain in "${domains[@]}"; do
    current=$((current + 1))
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "[${current}/${total_domains}] Processing: $domain (WITH CONTEXT)"
    echo "--------------------------------------------------------------------------------"
    
    python3 run_domain_gpt4o.py --domain "$domain" --use-context
    
    if [ $? -ne 0 ]; then
        echo "⚠️  WARNING: Failed to process $domain (with context)"
        echo "    Continuing to next domain..."
    else
        echo "✓ Successfully completed $domain (with context)"
    fi
    
    echo ""
done

# Reset counter for second phase
current=0

echo ""
echo "================================================================================"
echo "PHASE 2: Running WITHOUT CONTEXT (${total_domains} domains)"
echo "================================================================================"
echo ""

for domain in "${domains[@]}"; do
    current=$((current + 1))
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "[${current}/${total_domains}] Processing: $domain (NO CONTEXT)"
    echo "--------------------------------------------------------------------------------"
    
    python3 run_domain_gpt4o.py --domain "$domain" --no-context
    
    if [ $? -ne 0 ]; then
        echo "  WARNING: Failed to process $domain (no context)"
        echo "    Continuing to next domain..."
    else
        echo "✓ Successfully completed $domain (no context)"
    fi
    
    echo ""
done

# Calculate elapsed time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
elapsed_min=$((elapsed / 60))
elapsed_sec=$((elapsed % 60))

echo ""
echo "================================================================================"
echo "                           ALL DOMAINS COMPLETE!"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  • Processed: ${total_domains} domains"
echo "  • Total tasks: $((total_domains * 10 * 2)) (160 tasks × 2 runs)"
echo "  • Time elapsed: ${elapsed_min}m ${elapsed_sec}s"
echo ""
echo "Results saved to: results/<domain>/gpt4o_mini_{with/no}_context_results.csv"
echo ""
echo "Next steps:"
echo "  1. Run your classifier.py to generate comparison.csv files"
echo "  2. Compare GPT-4o Mini vs Llama 3.2-3B context usage"
echo "  3. Analyze results!"
echo ""
echo "================================================================================"