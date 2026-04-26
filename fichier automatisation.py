#!/usr/bin/env python
# coding: utf-8




import yfinance as yf
import pandas as pd
import glob
import os
from datetime import date





def mettre_a_jour_dossier(dossier="Companies_historical_data/"):
    fichiers = glob.glob(f"{dossier}/*.csv")
    
    for fichier in fichiers:
    
        ticker = os.path.basename(fichier).replace(".csv", "")
        
        df_existant   = pd.read_csv(fichier, index_col="Date", parse_dates=True)
        derniere_date = df_existant.index.max().strftime("%Y-%m-%d")
        aujourd_hui   = date.today().strftime("%Y-%m-%d")
        
        print(f"Mise à jour {ticker} depuis {derniere_date}...")
        
        # Télécharge les nouvelles données
        df_nouveau = yf.download(ticker, start=derniere_date, end=aujourd_hui)
        
        # Fusionner et sauvegarder
        df_final = pd.concat([df_existant, df_nouveau])
        df_final = df_final[~df_final.index.duplicated(keep="last")]
        df_final.sort_index(inplace=True)
        df_final.to_csv(fichier)
        
        print(f" {ticker} mis à jour jusqu'au {df_final.index.max().date()}")

mettre_a_jour_dossier()

