{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    nixos-wsl = { 
      url = "github:nix-community/NixOS-WSL/main";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, nixos-wsl, ... }: {
    
    nixosConfigurations = {
      
      nlin-EEG = nixpkgs.lib.nixosSystem {
        
        system = "x86_64-linux";
        
        modules = [
          ./nixosWSL.nix
          #{ nix.registry.nixpkgs.flake = nixpkgs; }
          nixos-wsl.nixosModules.default
          {
            system.stateVersion = "25.05";
            wsl.enable = true;
            wsl.defaultUser = "nixos";
            wsl.useWindowsDriver = true;
          }
        ];
      };
    };
  };
}
