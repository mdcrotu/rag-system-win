#!/usr/bin/env python3
"""
Script to help organize specification PDFs (Windows)
"""
import os
from pathlib import Path

def create_spec_structure():
    specs_dir = Path("specs")
    specs_dir.mkdir(exist_ok=True)
    
    # Create info file about where to get specs
    info_content = '''
# Hardware Specification Sources

## SystemVerilog (IEEE 1800)
- Source: IEEE Xplore Digital Library
- Search: "IEEE 1800-2017" or "IEEE 1800-2023"
- File: Save as 'ieee_1800_systemverilog.pdf'

## Ethernet (IEEE 802.3)
- Source: IEEE Xplore Digital Library
- Search: "IEEE 802.3-2022"
- File: Save as 'ieee_802_3_ethernet.pdf'

## USB Specification
- Source: USB Implementers Forum (usb.org)
- Document: "Universal Serial Bus Specification Revision 3.2"
- File: Save as 'usb_specification.pdf'

## PCIe Specification  
- Source: PCI-SIG (pcisig.com)
- Document: "PCI Express Base Specification Revision 6.0"
- File: Save as 'pcie_specification.pdf'

Note: Some specs require IEEE membership or purchase.
For testing, you can use any technical PDF document.
'''
    
    with open(specs_dir / "README.md", "w", encoding='utf-8') as f:
        f.write(info_content)
    
    print("Created specs directory structure")
    print("Please download your specification PDFs to the specs/ directory")
    print("See specs/README.md for download sources")

if __name__ == "__main__":
    create_spec_structure()
