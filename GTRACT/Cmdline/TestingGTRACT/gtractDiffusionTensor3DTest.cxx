/*=========================================================================

  This test was written by Joy Matsui to test the gtractDiffusionTensor3D 
  class to ensure that DTI scalar values were being computed correctly.

=========================================================================*/

#include<iostream>
#include<fstream>
#include<itkImage.h>
#include<itkImageFileReader.h>
#include<itkImageRegionIterator.h>
#include<itkImageRegionConstIterator.h>
#include "../Common/gtractDiffusionTensor3D.h"
#include "algo.h"
#include "GtractTypes.h"
#include "GenerateCLP.h"
#include "BRAINSThreadControl.h"
#include<itkDiffusionTensor3D.h>


int main( int argc, char *argv[] )
{
  typedef double TensorComponentType;
  typedef itk::gtractDiffusionTensor3D< TensorComponentType> TensorPixelType;
  typedef itk::Image< TensorPixelType, 3 > TensorImageType;

  
  // Read in tensor image designated as test data
  typedef itk::ImageFileReader< TensorImageType > TensorImageReaderType;
  TensorImageReaderType::Pointer tensorImageReader = TensorImageReaderType::New();
  tensorImageReader->SetFileName( argv[ 1 ] );
  
  TensorImageType::Pointer tensorImage = tensorImageReader->GetOutput();
  TensorImageType::IndexType tensorIndex;

  typedef itk::ImageRegionConstIterator< TensorImageType > ConstIteratorType;
  ConstIteratorType tensorIt( tensorImage, tensorImage->GetRequestedRegion() );

  for( tensorIt.GoToBegin(); !tensorIt.IsAtEnd(); ++tensorIt )
  {
    TensorPixelType tensorPixel = tensorIt.Get();
    tensorIndex = tensorIt.GetIndex();

    if( tensorIndex[ 0 ] == 31 && tensorIndex[ 1 ] == 1 && tensorIndex[ 2 ] == 0 )
    {
      // Test trace or ADC
      float testTraceADC = 0.0;
      testTraceADC = static_cast< float >( tensorPixel.GetTrace() / 3.0 );
      std::cout << "testTraceADC: " << testTraceADC << std::endl;
      if( tensorPixel.GetTrace() != 0.00130511 && testTraceADC != 0.000435037 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test FA
      float testFA = 0.0;
      testFA = static_cast< float >( tensorPixel.GetFractionalAnisotropy() );
      std::cout << "testFA: " << testFA << std::endl;
      if( testFA != 0.725469 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test RA
      float testRA = 0.0;
      testRA = static_cast< float >( tensorPixel.GetRelativeAnisotropy() );
      std::cout << "testFA: " << testRA << std::endl;
      if( testFA != 0.0201814 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test VR
      float testVR = 0.0;
      testVR = static_cast< float >( tensorPixel.GetVolumeRatio() );
      std::cout << "testVR: " << testVR << std::endl;
      if( testVR != 0.783673 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test AD
      float testAD = 0.0;
      testAD = static_cast< float >( tensorPixel.GetAxialDiffusivity() );
      std::cout << "testAD: " << testAD << std::endl;
      if( testAD != 0.000833842 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test RD
      float testRD = 0.0;
      testRD = static_cast< float >( tensorPixel.GetRadialDiffusivity() );
      std::cout << "testRD: " << testRD << std::endl;
      if( testRD != 0.000235635 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }
      
      // Test LI
      float testLI = 0.0;
      testLI = static_cast< float >( tensorPixel.GetLatticeIndex() );
      std::cout << "testLI: " << testLI << std::endl;
      if( testLI != 0.625887 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

    }

  }

}
