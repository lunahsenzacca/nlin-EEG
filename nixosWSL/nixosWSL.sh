#!/usr/bin/env bash

printf '\nChecking if the repository is in the correct place...\n'

if [ ! -d "$HOME/nlin-EEG" ]; then

  printf '\nRepository not found, downloading...\n'

  nix-shell -p git --command 'git clone https://github.com/lunahsenzacca/nlin-EEG.git $HOME/nlin-EEG'

  printf '\nDone!\n'

else
  printf '\nRepository FOUND, download skipped\n'
fi


printf '\nCopying important files into system directories...\n'

sudo cp $HOME/nlin-EEG/nixosWSL/flake.nix /etc/nixos
sudo cp $HOME/nlin-EEG/nixosWSL/nixosWSL.nix /etc/nixos
sudo cp $HOME/nlin-EEG/nixosWSL/.bash_profile $HOME/

printf '\nUpdating NixOS and installing system dependencies...\n'

sudo nixos-rebuild switch --flake '/etc/nixos#nlin-EEG'

printf '\nDone!\n'

printf '\nInstalling python dependencies...\n'

cd ~/nlin-EEG && devenv shell poetry install

printf '\nAll done! After the environment loads try running "menu"\n'

cd ~/nlin-EEG && devenv shell
