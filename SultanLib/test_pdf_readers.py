"""
Testing out common PDF reader libs in Python;
Common problems:
 - Lines are not fully read by lib, split in two, however visually in PDF doc its one line

Available libs:
    - PDFMiner
    - PyPDF2
    - Tabula-py
    - Slate
    - PDFQuery

Useful links:
https://medium.com/@umerfarooq_26378/python-for-pdf-ef0fac2808b0

"""

import unittest
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


class PDFReadersTestCase(unittest.TestCase):
    def test_PYPDF2(self):
        """
        Too many \n brake lines
        """
        import PyPDF2
        pdfFileObj = open('1.pdf', 'rb')
        # pdf reader object
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pageObj = pdfReader.getPage(0)
        print(pageObj.extractText())
        return_text = '''
        E
        DUCATION AND 
        Q
        UALIFICATIONS
         
        Royal Melbourne Institute of Technology (RMIT)
         
        Jul 2014 
        
           
        Oct 2015
         
        Bachelor of Business (Accountancy)
         
        GPA: 3.1 / 4
         
        Temasek Polytechnic
         
        Apr 2009 
        
         
        Apr 2012
         
        Diploma in Accounti
        ng and Finance
         
        Ngee Ann Secondary School
         
                    
        Jan 2005 
        
         
        Nov 2008
         
        
         
          
         
        S
        KILLS AND COMPETENCIES
         
        
         
        Trained in accounting softwares, Aexeo, Agresso Business World, Sage AccPac and MYOB
         
        
         
        Proficient in Microsoft Excel, PowerPoint 
        and Word
         
        
         
        Competent in Financial and Economic Databases (Bloomberg, Thomson Reuters)
         
        
         
        Languages spoken: English, Chinese, Cantonese
         
        
         
        Languages written: English, Chinese
        '''

    def test_tabula_py(self):
        """
        Piece of shit
        """
        import tabula
        # readinf the PDF file that contain Table Data
        # you can find find the pdf file with complete code in below
        # read_pdf will save the pdf table into Pandas Dataframe
        df = tabula.read_pdf("2.pdf")
        # in order to print first 5 lines of Table
        df.head()

    def test_pdf_miner_six(self):
        from pdfminer3.layout import LAParams, LTTextBox
        from pdfminer3.pdfpage import PDFPage
        from pdfminer3.pdfinterp import PDFResourceManager
        from pdfminer3.pdfinterp import PDFPageInterpreter
        from pdfminer3.converter import PDFPageAggregator
        from pdfminer3.converter import TextConverter
        import io

        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)

        with open('1.pdf', 'rb') as fh:
            for page in PDFPage.get_pages(fh,
                                          caching=True,
                                          check_extractable=True):
                page_interpreter.process_page(page)

            text = fake_file_handle.getvalue()

        # close open handles
        converter.close()
        fake_file_handle.close()

        print(text)
        return_text = '''       
        EDUCATION AND QUALIFICATIONS 

        Royal Melbourne Institute of Technology (RMIT) 
        
        Jul 2014 –   Oct 2015 
        
        Bachelor of Business (Accountancy) 
        
        GPA: 3.1 / 4 
        
        Temasek Polytechnic 
        
        Diploma in Accounting and Finance 
        
        Ngee Ann Secondary School 
        
        GCE ‘O’ Levels Certificate 
        
           
        
        SKILLS AND COMPETENCIES 
        
        Apr 2009 – Apr 2012 
        
                    Jan 2005 – Nov 2008 
        
        •  Trained in accounting softwares, Aexeo, Agresso Business World, Sage AccPac and MYOB 
        
        •  Proficient in Microsoft Excel, PowerPoint and Word 
        
        •  Competent in Financial and Economic Databases (Bloomberg, Thomson Reuters) 
        
        • 
        
        • 
        
        Languages spoken: English, Chinese, Cantonese 
        
        Languages written: English, Chinese 
        
        

        '''
    def test_fitz(self):
        import fitz
        doc = fitz.open('1.pdf')
        for page in doc:
            text = page.getText()
            print(text)
        '''
        EDUCATION AND QUALIFICATIONS 
        Royal Melbourne Institute of Technology (RMIT) 
        Jul 2014 –   Oct 2015 
        Bachelor of Business (Accountancy) 
        GPA: 3.1 / 4 
        Temasek Polytechnic 
        Apr 2009 – Apr 2012 
        Diploma in Accounting and Finance 
        Ngee Ann Secondary School 
                    Jan 2005 – Nov 2008 
        GCE ‘O’ Levels Certificate 
           
        SKILLS AND COMPETENCIES 
        • 
        Trained in accounting softwares, Aexeo, Agresso Business World, Sage AccPac and MYOB 
        • 
        Proficient in Microsoft Excel, PowerPoint and Word 
        • 
        Competent in Financial and Economic Databases (Bloomberg, Thomson Reuters) 
        • 
        Languages spoken: English, Chinese, Cantonese 
        • 
        Languages written: English, Chinese 
        '''
    def test_pdfminer(self):
        output = StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams(char_margin=20))
        interpreter = PDFPageInterpreter(manager, converter)
        pagenums = set()
        infile = open('1.pdf', 'rb')
        for page in PDFPage.get_pages(infile, pagenums):
            interpreter.process_page(page)
        infile.close()
        converter.close()
        text = output.getvalue()
        print(text)
        '''
        Process finished with exit code 0
        EDUCATION AND QUALIFICATIONS 
        
        Royal Melbourne Institute of Technology (RMIT) 
        Bachelor of Business (Accountancy) 
        GPA: 3.1 / 4 
        
        Temasek Polytechnic 
        Diploma in Accounting and Finance 
        
        Ngee Ann Secondary School 
        GCE ‘O’ Levels Certificate 
        
           
        
        SKILLS AND COMPETENCIES 
        
        Jul 2014 –   Oct 2015 
        
        Apr 2009 – Apr 2012 
        
                    Jan 2005 – Nov 2008 
        
        •  Trained in accounting softwares, Aexeo, Agresso Business World, Sage AccPac and MYOB 
        •  Proficient in Microsoft Excel, PowerPoint and Word 
        •  Competent in Financial and Economic Databases (Bloomberg, Thomson Reuters) 
        • 
        • 
        
        Languages spoken: English, Chinese, Cantonese 
        Languages written: English, Chinese 
                '''
