import React, { useRef } from 'react';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';

function PDFGenerator({
  imageState,
  segmentedImageState,
  category,
  description,
}) {
  const downloadPdf = () => {
    // Create a new jsPDF instance
    const pdf = new jsPDF();

    // Get the content of the component to be converted into PDF
    const pdfContent = document.getElementById('pdf-content');

    // Use html2canvas to capture the content and convert it into an image
    html2canvas(pdfContent).then((canvas) => {
      const imgData = canvas.toDataURL('image/png');

      // Add the captured image to the PDF
      pdf.addImage(imgData, 'PNG', 10, 10, 180, 180);

      // Add category and description text to the PDF
      pdf.text(category, 10, 200);
      pdf.text(description, 10, 210);

      // Save the PDF
      pdf.save('result.pdf');
    });
  };

  return (
    <div id='pdf-content'>
      {/* Render the content you want to capture in the PDF */}
      <img src={imageState} alt='Uploaded' />
      <img src={segmentedImageState} alt='Segmented' />
      <p>{category}</p>
      <p>{description}</p>
      <button onClick={downloadPdf}>Download PDF</button>
    </div>
  );
}

export default PDFGenerator;
