#!/usr/bin/env bash

if [ ! -d "/$HOME/nlin-EEG" ]; then
  nix-shell -p git --command 'git clone https://github.com/lunahsenzacca/nlin-EEG.git /$HOME/nlin-EEG'
fi

#sudo nix-channel --update

sudo cp /$HOME/nlin-EEG/nixosWSL/flake.nix /etc/nixos
sudo cp /$HOME/nlin-EEG/nixosWSL/nixosWSL.nix /etc/nixos

sudo nixos-rebuild switch --flake "/etc/nixos#nlin-EEG" --impure

cd ~/nlin-EEG && devenv shell poetry install

cd ~/nlin-EEG && devenv shell
