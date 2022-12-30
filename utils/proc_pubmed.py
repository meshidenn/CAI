import argparse
import os
import gzip
from xml.etree import ElementTree as ET
from glob import glob
from tqdm import tqdm


def extract_abst(xml_files, out_path, out_path_clean):
    with open(out_path, "w") as fout:
        with open(out_path_clean, "w") as fout_clean:
            for xml_file in tqdm(xml_files):
                with gzip.open(xml_file, "rt", encoding="utf-8") as fin:
                    read_xml = fin.read()
                    root = ET.fromstring(read_xml)
                    for i, e in enumerate(root.iter()):
                        if e.tag == "AbstractText":
                            if e.text is not None:
                                print(e.text, file=fout)
                                if len(e.text.split()) > 127:
                                    print(e.text, file=fout_clean)


def main(args):
    pubmed_xml_files = sorted(glob(os.path.join(args.root_dir, "pubmed22*.xml.gz")))
    extract_abst(pubmed_xml_files, args.out_path, args.out_path_clean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir")
    parser.add_argument("--out_path")
    parser.add_argument("--out_path_clean")

    args = parser.parse_args()
    main(args)
