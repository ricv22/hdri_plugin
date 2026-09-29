#!/bin/bash
# Double-click on your Mac. Opens the composer in your browser. No server needed.
cd "$(dirname "$0")"
FILE="$(pwd)/composer.html"
if [ ! -f "$FILE" ]; then
  echo "Chybi composer.html. Stahni aktualni vetvi cursor/expand-production-contacts-c02a."
  read -r -p "Stiskni Enter pro zavreni..."
  exit 1
fi
echo "Oteviram composer v prohlizeci..."
echo "$FILE"
if command -v open >/dev/null 2>&1; then
  open "$FILE"
else
  echo "Otevri ten soubor rucne v Chrome nebo Safari."
fi
sleep 2
