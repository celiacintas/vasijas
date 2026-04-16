import argparse
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem

def extract_images(input_doc_path, output_dir):
    """Extract images from PDF document."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2  # Adjust image resolution if needed
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    conv_res = doc_converter.convert(input_doc_path)
    doc_filename = conv_res.input.file.stem

    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            img_path = output_dir / f"{doc_filename}-table-{table_counter}.png"
            element.get_image(conv_res.document).save(img_path, "PNG")
            print(f"Saved: {img_path}")
        if isinstance(element, PictureItem):
            picture_counter += 1
            img_path = output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            element.get_image(conv_res.document).save(img_path, "PNG")
            print(f"Saved: {img_path}")
    
    print(f"\nExtraction complete: {table_counter} tables, {picture_counter} pictures")

def main():
    parser = argparse.ArgumentParser(
        description="Extract images (tables and pictures) from PDF documents using Docling."
    )
    
    parser.add_argument(
        "--pdf",
        type=str,
        default="data/thesis_texture.pdf",
        help="Path to the input PDF file (default: data/thesis_texture.pdf)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output_images",
        help="Path to the output directory for extracted images (default: output_images)"
    )
    
    args = parser.parse_args()
    
    input_doc_path = Path(args.pdf)
    output_dir = Path(args.output)
    
    # Validate input file exists
    if not input_doc_path.exists():
        print(f"Error: Input PDF file not found: {input_doc_path}")
        return
    
    if not input_doc_path.suffix.lower() == ".pdf":
        print(f"Error: Input file must be a PDF: {input_doc_path}")
        return
    
    print(f"Processing PDF: {input_doc_path}")
    print(f"Output directory: {output_dir}")
    
    extract_images(input_doc_path, output_dir)

if __name__ == "__main__":
    main()

#Using default paths
#python extract_images.py

# Custom PDF path
#python extract_images.py --pdf path/to/your/document.pdf

# Custom output directory
#python extract_images.py --output extracted_figures

# Both custom paths
#python extract_images.py --pdf data/thesis_texture.pdf --output output_images

# Help
#python extract_images.py --help