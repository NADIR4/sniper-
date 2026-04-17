"""Univers d'actions suivies : ~300 actions USA + Europe pour entraînement maximal."""
from __future__ import annotations

# === USA : S&P 500 Top 150 + growth/tech extras ===
USA_TOP = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "LLY",
    "AVGO", "JPM", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ",
    "NFLX", "ABBV", "BAC", "CRM", "CVX", "MRK", "KO", "ORCL", "PEP", "AMD",
    "ADBE", "WMT", "TMO", "LIN", "CSCO", "ACN", "MCD", "ABT", "DIS", "WFC",
    "NOW", "IBM", "TXN", "CAT", "VZ", "GE", "INTU", "AMGN", "AXP", "PFE",
    "ISRG", "QCOM", "DHR", "BKNG", "GS", "BX", "T", "SYK", "MS", "ELV",
    "NEE", "PGR", "PM", "LOW", "BSX", "SPGI", "TJX", "SCHW", "C", "UBER",
    "ADP", "HON", "VRTX", "DE", "CB", "REGN", "BLK", "AMAT", "MMC", "FI",
    "MU", "ETN", "PLD", "BMY", "ADI", "PANW", "COP", "LMT", "MDT", "ICE",
    "SO", "KKR", "CI", "UPS", "TT", "SBUX", "DUK", "CMG", "AMT", "GILD",
    "EQIX", "SHW", "CME", "WM", "MO", "AON", "ANET", "ITW", "MDLZ", "CVS",
    "GD", "MCO", "CL", "APH", "TGT", "ZTS", "WELL", "BDX", "CDNS", "USB",
    "NOC", "PH", "MMM", "EMR", "SNPS", "CRWD", "RCL", "CSX", "COF", "MSI",
    "ORLY", "PYPL", "DELL", "CMCSA", "HCA", "APD", "FCX", "NSC", "PLTR", "ECL",
    "WMB", "SPG", "AJG", "TFC", "CEG", "AZO", "AFL", "ADSK", "OXY", "NXPI",
    "TRV", "EOG", "MRVL", "KLAC", "LRCX", "ROP", "PSX", "FDX", "CTAS", "SLB",
    # Growth / mid-cap opportunities
    "SMCI", "ARM", "COIN", "MSTR", "SHOP", "SNOW", "DDOG", "NET", "CRWD", "ZS",
    "ABNB", "RBLX", "U", "SOFI", "HOOD", "AFRM", "RIVN", "LCID", "PLUG", "BE",
    "ENPH", "FSLR", "SEDG", "CHPT", "NKLA", "RKLB", "ASTS", "JOBY", "ACHR", "BBAI",
]

# === EUROPE : Euro Stoxx 600 subset + UK ===
EUROPE_TOP = [
    # Pays-Bas
    "ASML.AS", "PRX.AS", "ADYEN.AS", "INGA.AS", "PHIA.AS", "AD.AS", "HEIA.AS",
    "REN.AS", "WKL.AS", "MT.AS", "RAND.AS", "DSFIR.AS", "AKZA.AS", "KPN.AS",
    # France
    "MC.PA", "OR.PA", "RMS.PA", "SAN.PA", "AIR.PA", "BNP.PA", "TTE.PA", "SU.PA",
    "AI.PA", "CS.PA", "KER.PA", "DG.PA", "EL.PA", "BN.PA", "RI.PA", "SGO.PA",
    "STLA.PA", "ENGI.PA", "VIE.PA", "LR.PA", "ACA.PA", "CAP.PA", "GLE.PA",
    "ML.PA", "PUB.PA", "SAF.PA", "ATO.PA", "TEP.PA", "HO.PA",
    # Allemagne
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "BAYN.DE", "BMW.DE", "DBK.DE",
    "IFX.DE", "MBG.DE", "MUV2.DE", "VOW3.DE", "ADS.DE", "RWE.DE", "DB1.DE", "HEN3.DE",
    "EOAN.DE", "MRK.DE", "FME.DE", "DHL.DE", "BEI.DE", "CON.DE", "SRT3.DE",
    "ZAL.DE", "HNR1.DE", "BNR.DE", "1COV.DE",
    # Suisse
    "NESN.SW", "NOVN.SW", "ROG.SW", "CFR.SW", "UBSG.SW", "ZURN.SW", "ABBN.SW",
    "GIVN.SW", "SGSN.SW", "LONN.SW", "HOLN.SW", "GEBN.SW", "SIKA.SW", "SCMN.SW",
    # Italie
    "RACE.MI", "UCG.MI", "ISP.MI", "ENI.MI", "ENEL.MI", "STM.MI", "TIT.MI",
    "G.MI", "MB.MI", "BAMI.MI", "BPE.MI", "FCT.MI",
    # Espagne
    "BBVA.MC", "SAN.MC", "IBE.MC", "ITX.MC", "TEF.MC", "REP.MC", "AENA.MC",
    "FER.MC", "ACS.MC", "ELE.MC", "MAP.MC",
    # UK (London)
    "AZN.L", "HSBA.L", "SHEL.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "DGE.L",
    "GLEN.L", "REL.L", "BATS.L", "VOD.L", "PRU.L", "BARC.L", "LLOY.L",
    "NWG.L", "STAN.L", "CPG.L", "EXPN.L", "AAL.L", "LSEG.L",
    # Nordics
    "NOVO-B.CO", "VWS.CO", "DSV.CO", "CARL-B.CO", "ORSTED.CO",
    "ERIC-B.ST", "VOLV-B.ST", "INVE-B.ST", "ATCO-B.ST", "SAND.ST",
    "EQNR.OL", "DNB.OL", "MOWI.OL",
]

UNIVERSE = USA_TOP + EUROPE_TOP


def get_universe() -> list[str]:
    return list(dict.fromkeys(UNIVERSE))


SP500_TOP50 = USA_TOP[:50]
EUROSTOXX_TOP50 = EUROPE_TOP[:50]
