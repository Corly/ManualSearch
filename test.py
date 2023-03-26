

def try_pypdf2(path):
    import PyPDF2
    
    reader = PyPDF2.PdfReader(path)
    print(reader.getNumPages())
    for page in reader.pages:
        print(page.extractText())
    

def try_fitz(path):
    import fitz
    from PIL import Image
    import io
    
    doc = fitz.open(path)
    print(len(doc))
    for page in doc:
        print(page.get_text('text'))
    
    blocks = doc[0].get_text('blocks')
    print(blocks)
    print()
    for block in blocks:
        print(block, block[-1])
        if block[-1] == 0:
            print(block[4])
            print()
            
    # iterate over pdf pages
    for page_index in range(len(doc)):
        # get the page itself
        page = doc[page_index]
        image_list = page.get_images()
        # printing number of images found in this page
        if image_list:
            print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        else:
            print("[!] No images found on page", page_index)
        for image_index, img in enumerate(page.get_images(), start=1):
            # get the XREF of the image
            print(img)
            xref = img[0]
    
            # extract the image bytes
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # get the image extension
            image_ext = base_image["ext"]
            # load it to PIL
            image = Image.open(io.BytesIO(image_bytes))
            # save it to local disk
            image.save(open(f"image{page_index+1}_{image_index}.{image_ext}", "wb"))


def test_extract_tables(path):
    from tabula import read_pdf
    from tabulate import tabulate 

    df = read_pdf(path,pages="all") #address of pdf file
    print(df)


if __name__ == '__main__':
    # try_pypdf2("test.pdf")
    try_fitz("test_tabel.pdf")
    # test_extract_tables("test.pdf")