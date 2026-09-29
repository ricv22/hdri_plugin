#!/bin/bash
# Double-click this file on your Mac. It starts the composer and opens a browser.
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 neni nainstalovany."
  echo "Stahni ho z https://www.python.org/downloads/ a spust tento soubor znovu."
  if command -v open >/dev/null 2>&1; then
    open "https://www.python.org/downloads/"
  fi
  read -r -p "Stiskni Enter pro zavreni..."
  exit 1
fi
echo
echo "Spoustim outreach composer..."
echo "Prohlizec by se mel otevrit sam. Jinak otevri: http://127.0.0.1:8765"
echo "Prihlaseni Gmailem neni nutne — maily muzes kopirovat."
echo
exec python3 contact_app.py "$@"
