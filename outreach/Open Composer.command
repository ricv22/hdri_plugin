#!/bin/bash
# Double-click on the Mac. Starts the composer on this computer and opens Safari.
cd "$(dirname "$0")"
xattr -d com.apple.quarantine "$0" 2>/dev/null || true

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 neni nainstalovany."
  echo "Stahni ho z https://www.python.org/downloads/ a spust tento soubor znovu."
  if command -v open >/dev/null 2>&1; then
    open "https://www.python.org/downloads/"
  fi
  read -r -p "Stiskni Enter pro zavreni..."
  exit 1
fi

echo "Spoustim composer na tomhle Macu."
echo "Safari se otevre na http://127.0.0.1:8765"
echo "Tohle okno nech otevrene."
echo
exec python3 contact_app.py --host 127.0.0.1 --port 8765
