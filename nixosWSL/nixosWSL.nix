{ config, lib, pkgs, ... }:

{
  environment.systemPackages = with pkgs; [
    devenv
    direnv
  ];

  fonts.packages = with pkgs; [
    jetbrains-mono
    nerd-fonts.jetbrains-mono
  ];

  nix.settings.experimental-features = [ "nix-command" "flakes" ];

  nix.settings.trusted-users = [ "root" "nixos" ];

  system.stateVersion = "25.05";
}
