"""Affiche le top 10 des scores actuels (même sous le seuil)."""
from ml.scanner import scan_market

results = scan_market()
print(f"\n=== TOP 15 OPPORTUNITÉS ACTUELLES ===\n")
print(f"{'Rang':<5}{'Ticker':<12}{'Score':<10}{'RF':<8}{'XGB':<8}{'Iso':<8}{'Prix':<10}")
print("-" * 60)
for i, r in enumerate(results[:15], 1):
    flag = "🎯" if r.consensus >= 0.75 else "  "
    print(f"{flag}{i:<3}{r.ticker:<12}{r.consensus:.1%}    {r.rf_prob:.0%}   {r.xgb_prob:.0%}   {r.iso_score:.0%}   {r.price:.2f}")
