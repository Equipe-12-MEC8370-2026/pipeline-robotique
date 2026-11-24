@echo off
rem Ouvre l'interface CTA dans le navigateur (adresse locale, rien n'est publie sur Internet).
rem Fermer cette fenetre arrete l'interface ; les calculs deja lances continuent.
cd /d "%~dp0app"
"%~dp0.venv-app\Scripts\python.exe" -m streamlit run main.py
