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

  /*====================================================================

    Test #1: Test scalars with actual tensor file by computing scalars
    from selected voxel (31, 1, 0) whose scalars were pre-computed in
    MATLAB

  ====================================================================*/

  // Read in tensor image designated as test data
  typedef itk::ImageFileReader< TensorImageType > TensorImageReaderType;
  TensorImageReaderType::Pointer actualTensorImageReader = TensorImageReaderType::New();
  actualTensorImageReader->SetFileName( argv[ 1 ] );
  
  TensorImageType::Pointer actualTensorImage = actualTensorImageReader->GetOutput();
  TensorImageType::IndexType actualTensorIndex;

  typedef itk::ImageRegionConstIterator< TensorImageType > ConstIteratorType;
  ConstIteratorType actualTensorIt( actualTensorImage, actualTensorImage->GetRequestedRegion() );

  for( actualTensorIt.GoToBegin(); !actualTensorIt.IsAtEnd(); ++actualTensorIt )
  {
    TensorPixelType tensorPixel = actualTensorIt.Get();
    actualTensorIndex = actualTensorIt.GetIndex();

    if( actualTensorIndex[ 0 ] == 31 && actualTensorIndex[ 1 ] == 1 && actualTensorIndex[ 2 ] == 0 )
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
  
  /*====================================================================

    Test #2: Test scalars with synthetic tensor file that has values in
    its diagonal without crossterms.

  ====================================================================*/

  TensorImageType::IndexType start;
  start.Fill( 0 );

  TensorImageType::SizeType size;
  size.Fill( 11 );

  TensorImageType::RegionType simpleSyntheticTensorRegion( start, size );

  TensorImageType::Pointer simpleSyntheticTensor = TensorImageType::New();
  simpleSyntheticTensor->SetRegions( simpleSyntheticTensorRegion );
  simpleSyntheticTensor->Allocate();

  TensorImageType::IndexType simpleSyntheticTensorIndex;
    
  // Loop over all voxels and give them tensor values
  ConstIteratorType simpleSyntheticTensorIt( simpleSyntheticTensor, simpleSyntheticTensor->GetRequestedRegion() );

  for( simpleSyntheticTensorIt.GoToBegin(); !simpleSyntheticTensorIt.IsAtEnd(); ++simpleSyntheticTensorIt )
  {
    simpleSyntheticTensorIndex = simpleSyntheticTensorIt.GetIndex();
    TensorPixelType simpleSyntheticTensorPixel;
    simpleSyntheticTensorPixel[ 0 ] = 0.0008;
    simpleSyntheticTensorPixel[ 1 ] = 0.0;
    simpleSyntheticTensorPixel[ 2 ] = 0.0;
    simpleSyntheticTensorPixel[ 3 ] = 0.0007;
    simpleSyntheticTensorPixel[ 4 ] = 0.0;
    simpleSyntheticTensorPixel[ 5 ] = 0.0004;

    simpleSyntheticTensor->SetPixel( simpleSyntheticTensorIndex, simpleSyntheticTensorPixel );
  }

  for( simpleSyntheticTensorIt.GoToBegin(); !simpleSyntheticTensorIt.IsAtEnd(); ++simpleSyntheticTensorIt )
  {
    simpleSyntheticTensorIndex = simpleSyntheticTensorIt.GetIndex();
    TensorPixelType simpleSyntheticTensorPixel;

      // Test trace and ADC
      float testTraceADC = 0.0;
      testTraceADC = static_cast< float >( simpleSyntheticTensorPixel.GetTrace() / 3.0 );
      std::cout << "testTraceADC: " << testTraceADC << std::endl;
      if( simpleSyntheticTensorPixel.GetTrace() != 0.001900000 && testTraceADC != 0.000633333 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test FA
      float testFA = 0.0;
      testFA = static_cast< float >( simpleSyntheticTensorPixel.GetFractionalAnisotropy() );
      std::cout << "testFA: " << testFA << std::endl;
      if( testFA != 0.317451 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test RA
      float testRA = 0.0;
      testRA = static_cast< float >( simpleSyntheticTensorPixel.GetRelativeAnisotropy() );
      std::cout << "testFA: " << testRA << std::endl;
      if( testFA != 0.448944 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test VR
      float testVR = 0.0;
      testVR = static_cast< float >( simpleSyntheticTensorPixel.GetVolumeRatio() );
      std::cout << "testVR: " << testVR << std::endl;
      if( testVR != 0.118239 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test AD
      float testAD = 0.0;
      testAD = static_cast< float >( simpleSyntheticTensorPixel.GetAxialDiffusivity() );
      std::cout << "testAD: " << testAD << std::endl;
      if( testAD != 0.0008 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test RD
      float testRD = 0.0;
      testRD = static_cast< float >( simpleSyntheticTensorPixel.GetRadialDiffusivity() );
      std::cout << "testRD: " << testRD << std::endl;
      if( testRD != 0.00055 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }
      
      // Test LI
      float testLI = 0.0;
      testLI = static_cast< float >( simpleSyntheticTensorPixel.GetLatticeIndex() );
      std::cout << "testLI: " << testLI << std::endl;
      if( testLI != 0.015996 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }
    
   
  
  }

  /*====================================================================

    Test #3: Test scalars with synthetic tensor file that has 
    crossterms.

  ====================================================================*/

  TensorImageType::RegionType compSyntheticTensorRegion( start, size );

  TensorImageType::Pointer compSyntheticTensor = TensorImageType::New();
  compSyntheticTensor->SetRegions( compSyntheticTensorRegion );
  compSyntheticTensor->Allocate();

  TensorImageType::IndexType compSyntheticTensorIndex;
    
  // Loop over all voxels and give them tensor values
  ConstIteratorType compSyntheticTensorIt( compSyntheticTensor, compSyntheticTensor->GetRequestedRegion() );

  for( compSyntheticTensorIt.GoToBegin(); !compSyntheticTensorIt.IsAtEnd(); ++compSyntheticTensorIt )
  {
    compSyntheticTensorIndex = compSyntheticTensorIt.GetIndex();
    TensorPixelType compSyntheticTensorPixel;
    compSyntheticTensorPixel[ 0 ] = 0.0008;
    compSyntheticTensorPixel[ 1 ] = 0.0;
    compSyntheticTensorPixel[ 2 ] = 0.0;
    compSyntheticTensorPixel[ 3 ] = 0.0007;
    compSyntheticTensorPixel[ 4 ] = 0.0;
    compSyntheticTensorPixel[ 5 ] = 0.0004;

    compSyntheticTensor->SetPixel( compSyntheticTensorIndex, compSyntheticTensorPixel );
  }

  for( compSyntheticTensorIt.GoToBegin(); !compSyntheticTensorIt.IsAtEnd(); ++compSyntheticTensorIt )
  {
    compSyntheticTensorIndex = compSyntheticTensorIt.GetIndex();
    TensorPixelType compSyntheticTensorPixel;

    if( compSyntheticTensorIndex[ 0 ] == 5 && compSyntheticTensorIndex[ 1 ] == 5 && compSyntheticTensorIndex[ 2 ] == 0 )
    {
      // Test trace and ADC
      float testTraceADC = 0.0;
      testTraceADC = static_cast< float >( compSyntheticTensorPixel.GetTrace() / 3.0 );
      std::cout << "testTraceADC: " << testTraceADC << std::endl;
      if( compSyntheticTensorPixel.GetTrace() != 0.001900000 && testTraceADC != 0.000633333 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test FA
      float testFA = 0.0;
      testFA = static_cast< float >( compSyntheticTensorPixel.GetFractionalAnisotropy() );
      std::cout << "testFA: " << testFA << std::endl;
      if( testFA != 0.317451 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test RA
      float testRA = 0.0;
      testRA = static_cast< float >( compSyntheticTensorPixel.GetRelativeAnisotropy() );
      std::cout << "testFA: " << testRA << std::endl;
      if( testFA != 0.448944 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test VR
      float testVR = 0.0;
      testVR = static_cast< float >( compSyntheticTensorPixel.GetVolumeRatio() );
      std::cout << "testVR: " << testVR << std::endl;
      if( testVR != 0.118239 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test AD
      float testAD = 0.0;
      testAD = static_cast< float >( compSyntheticTensorPixel.GetAxialDiffusivity() );
      std::cout << "testAD: " << testAD << std::endl;
      if( testAD != 0.0008 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }

      // Test RD
      float testRD = 0.0;
      testRD = static_cast< float >( compSyntheticTensorPixel.GetRadialDiffusivity() );
      std::cout << "testRD: " << testRD << std::endl;
      if( testRD != 0.00055 )
      {
        return EXIT_SUCCESS;
      }
      else
      {
        return EXIT_FAILURE;
      }
      
      // Test LI
      float testLI = 0.0;
      testLI = static_cast< float >( compSyntheticTensorPixel.GetLatticeIndex() );
      std::cout << "testLI: " << testLI << std::endl;
      if( testLI != 0.015996 )
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
