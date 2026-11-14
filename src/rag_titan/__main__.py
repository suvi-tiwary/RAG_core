"""
Run the command line tool as `python -m rag_titan ...`.

Same as the `rag-titan` command. Useful on Windows when security settings
(Smart App Control, application-control policies) block the unsigned
`rag-titan.exe` launcher that pip creates.

    python -m rag_titan --version
    python -m rag_titan evaluate --input data.json --output report.json --k 3
"""

from rag_titan.cli import main

if __name__ == "__main__":
    main()
