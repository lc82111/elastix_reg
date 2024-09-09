#include "itkElastixRegistrationMethod.h"
#include "elxParameterObject.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkExtractImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkFlipImageFilter.h"
#include "itkThresholdImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImportImageFilter.h"
#include "itkRawImageIO.h" 
#include "itkRescaleIntensityImageFilter.h" // Add this line
#include <chrono>
#include <iostream>

// Define image type
using PixelType = float;
constexpr unsigned int Dimension = 3;
using ImageType = itk::Image<PixelType, Dimension>;

template<typename ImageType> void saveSliceToPNG(const typename ImageType::Pointer& img, const unsigned int sliceIndex, const std::string& outputFilename)
{
    using SliceType = itk::Image<typename ImageType::PixelType, 2>;
    using ExtractFilterType = itk::ExtractImageFilter<ImageType, SliceType>;
    using UCharSliceType = itk::Image<unsigned char, 2>;
    using WriterType = itk::ImageFileWriter<UCharSliceType>;

    // Extract slice from img
    typename ImageType::RegionType inputRegion = img->GetLargestPossibleRegion();
    typename ImageType::SizeType inputSize = inputRegion.GetSize();
    typename ImageType::IndexType inputStart = inputRegion.GetIndex();

    typename ImageType::RegionType desiredRegion;
    typename ImageType::SizeType desiredSize;
    typename ImageType::IndexType desiredStart;

    // Set up the extraction region
    for (unsigned int i = 0; i < ImageType::ImageDimension; ++i)
    {
        desiredStart[i] = inputStart[i];
        desiredSize[i] = inputSize[i];
    }
    desiredSize[1] = 0;  // We want to extract a 2D slice
    desiredStart[1] = sliceIndex;

    desiredRegion.SetSize(desiredSize);
    desiredRegion.SetIndex(desiredStart);

    std::cout << "Input region: " << inputRegion << std::endl;
    std::cout << "Desired region: " << desiredRegion << std::endl;
    std::cout << "Slice index: " << sliceIndex << std::endl;

    auto extractFilter = ExtractFilterType::New();
    extractFilter->SetExtractionRegion(desiredRegion);
    extractFilter->SetInput(img);
    extractFilter->SetDirectionCollapseToSubmatrix();

    // Rescale the intensity of the slice to unsigned char
    using RescaleFilterType = itk::RescaleIntensityImageFilter<SliceType, UCharSliceType>;
    auto rescaleFilter = RescaleFilterType::New();
    rescaleFilter->SetInput(extractFilter->GetOutput());
    rescaleFilter->SetOutputMinimum(0);
    rescaleFilter->SetOutputMaximum(255);

    auto writer = WriterType::New();
    writer->SetFileName(outputFilename);
    writer->SetInput(rescaleFilter->GetOutput());

    // Write the slice to a PNG file
    try {
        writer->Update();
    } catch (itk::ExceptionObject &error) {
        std::cerr << "Error: " << error << std::endl;
    }
}

ImageType::Pointer moveOriginToCenter(const ImageType::Pointer& image)
{
    // Get the size and spacing of the image
    ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
    ImageType::SpacingType spacing = image->GetSpacing();

    // Calculate the center of the image
    ImageType::PointType center;
    for (unsigned int i = 0; i < Dimension; ++i)
    {
        center[i] = size[i] * spacing[i] / 2.0;
    }

    // Set the origin of the image to the center
    image->SetOrigin(center);

    return image;
}

ImageType::Pointer loadITKImage(const std::string& filename)
{
    using ReaderType = itk::ImageFileReader<ImageType>;
    auto reader = ReaderType::New();
    reader->SetFileName(filename);
    reader->Update();
    return reader->GetOutput();
}

template<typename T> ImageType::Pointer loadRawImage3(const std::string& filename, const ImageType::SizeType& size, const ImageType::SpacingType& spacing)
{
    // Read the entire file into a vector
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file: " + filename);

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check if the file size matches the expected size
    size_t expectedSize = size[0] * size[1] * size[2] * sizeof(T);
    if (fileSize != expectedSize)
        throw std::runtime_error("File size does not match expected size. Expected: " + 
                                 std::to_string(expectedSize) + ", Actual: " + std::to_string(fileSize));

    std::vector<T> buffer(fileSize / sizeof(T));
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

    // Convert to float
    std::vector<float> floatBuffer(buffer.size());
    std::transform(buffer.begin(), buffer.end(), floatBuffer.begin(), 
                    [](T val) { return static_cast<float>(val); });

    // Create and populate the ITK image
    auto image = ImageType::New();
    image->SetRegions(size);
    image->SetSpacing(spacing);
    image->Allocate();
    // set the origin to the center
    // Calculate the center of the image
    ImageType::PointType center;
    for (unsigned int i = 0; i < Dimension; ++i)
        center[i] = size[i] * spacing[i] / 2.0;
    image->SetOrigin(center);

    // Copy the data into the ITK image
    std::copy(floatBuffer.begin(), floatBuffer.end(), image->GetBufferPointer());

    return image;
}

ImageType::Pointer flipImageAlongZAxis(const ImageType::Pointer& inputImage)
{
    using FlipFilterType = itk::FlipImageFilter<ImageType>; //
    auto flipFilter = FlipFilterType::New();
    flipFilter->SetInput(inputImage);
    FlipFilterType::FlipAxesArrayType flipAxes;
    flipAxes.Fill(false);
    flipAxes[2] = true;
    flipFilter->SetFlipAxes(flipAxes);
    flipFilter->Update();
    return flipFilter->GetOutput();
}

ImageType::Pointer thresholdImage(const ImageType::Pointer& inputImage, PixelType threshold)
{
    using ThresholdFilterType = itk::ThresholdImageFilter<ImageType>;
    auto thresholdFilter = ThresholdFilterType::New();
    thresholdFilter->SetInput(inputImage);
    thresholdFilter->ThresholdBelow(threshold);
    thresholdFilter->SetOutsideValue(0);
    thresholdFilter->Update();
    return thresholdFilter->GetOutput();
}

ImageType::Pointer resampleImage(const ImageType::Pointer& inputImage, const ImageType::SpacingType& outputSpacing, const ImageType::SizeType& outputSize)
{
    using ResampleFilterType = itk::ResampleImageFilter<ImageType, ImageType>;
    auto resampleFilter = ResampleFilterType::New();
    resampleFilter->SetInput(inputImage);
    resampleFilter->SetOutputSpacing(outputSpacing);
    resampleFilter->SetSize(outputSize);
    resampleFilter->SetOutputDirection(inputImage->GetDirection());
    resampleFilter->SetOutputOrigin(inputImage->GetOrigin());
    resampleFilter->SetInterpolator(itk::LinearInterpolateImageFunction<ImageType, double>::New());
    resampleFilter->Update();
    return resampleFilter->GetOutput();
}

void printImageInfo(const ImageType::Pointer& image, const std::string& label)
{
    std::cout << label << std::endl;
    std::cout << "Region: " << image->GetLargestPossibleRegion().GetSize() << " "
              << "Spacing: " << image->GetSpacing() << " " 
              << "Origin: " << image->GetOrigin() << " "
              << "Direction: " << image->GetDirection() << std::endl;
}

ImageType::Pointer load_ct()
{
    // Load CT image
    ImageType::SizeType ctSize = {512, 512, 144};
    ImageType::SpacingType spacing;
    spacing[0] = 0.47;
    spacing[1] = 0.47;
    spacing[2] = 1.47;
    auto ctImage = loadRawImage3<float>("/home/congliu/linatech/cbct_correct_niu2010/CT_512_512_144.raw", ctSize, spacing);
    printImageInfo(ctImage, "CT Image");

    // Process CT image
    ctImage = flipImageAlongZAxis(ctImage);
    ctImage = thresholdImage(ctImage, 0.01);
    printImageInfo(ctImage, "CT Image (processed)");
    
    // Resample CT image
    ImageType::SpacingType outputSpacing;
    outputSpacing[0] = 0.5;
    outputSpacing[1] = 0.5;
    outputSpacing[2] = 0.5;
    ImageType::SizeType outputSize = {512, 512, static_cast<unsigned int>(144 * 1.47 / 0.5)};
    ctImage = resampleImage(ctImage, outputSpacing, outputSize);
    printImageInfo(ctImage, "Resampled CT Image");

    // Move the origin of the image to the center
    ctImage = moveOriginToCenter(ctImage);
    printImageInfo(ctImage, "Centered CT Image");

    // save image to disk
    using WriterType = itk::ImageFileWriter<ImageType>;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName("ct_image.nii");
    writer->SetInput(ctImage);
    writer->Update();
    return ctImage;
}

ImageType::Pointer load_cbct()
{
    // Load CBCT image
    ImageType::SizeType cbctSize = {512, 512, 512};
    ImageType::SpacingType spacing;
    spacing[0] = 0.5;
    spacing[1] = 0.5;
    spacing[2] = 0.5;
    auto cbctImage = loadRawImage3<float>("/home/congliu/linatech/cbct_correct_niu2010/CBCT.raw", cbctSize, spacing);
    printImageInfo(cbctImage, "CBCT Image");

    // Move the origin of the image to the center
    cbctImage = moveOriginToCenter(cbctImage);
    printImageInfo(cbctImage, "Centered CBCT Image");

    // save the images to disk
    using WriterType = itk::ImageFileWriter<ImageType>;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName("cbct_image.nii");
    writer->SetInput(cbctImage);
    writer->Update();
    return cbctImage;
}

int executeElastix(const std::string& fixedImagePath, const std::string& movingImagePath, const std::string& outputPath, const std::string& parameterFilePath)
{
    std::ostringstream cmd;
    cmd << "elastix"
        << " -f " << fixedImagePath
        << " -m " << movingImagePath
        << " -out " << outputPath
        << " -p " << parameterFilePath;
    
    std::cout << "Executing command: " << cmd.str() << std::endl;
    return std::system(cmd.str().c_str());
}

void libElastix(ImageType::Pointer fixedImage, ImageType::Pointer movingImage, bool useOpenCL)
{
    // Use elastix library
    using ElastixType = itk::ElastixRegistrationMethod<ImageType, ImageType>;
    ElastixType::Pointer elastix = ElastixType::New();
    elastix->LogToConsoleOn();

    elastix->SetFixedImage(fixedImage);
    elastix->SetMovingImage(movingImage);

    auto parameterObject = ElastixType::ParameterObjectType::New();
    auto parameterMap = ElastixType::ParameterObjectType::GetDefaultParameterMap("rigid");
    parameterMap["MaximumNumberOfIterations"] = std::vector<std::string>{"600"};
    parameterMap["NumberOfResolutions"] = std::vector<std::string>{"3"};

    if (useOpenCL)
    {
        parameterMap["UseOpenCL"] = std::vector<std::string>{"true"};
        parameterMap["OpenCLDeviceType"] = std::vector<std::string>{"GPU"};
        parameterMap["OpenCLDeviceIndex"] = std::vector<std::string>{"0"};
        parameterMap["FixedImagePyramid"] = std::vector<std::string>{"OpenCLFixedGenericImagePyramid"};
        parameterMap["OpenCLFixedGenericImagePyramidUseOpenCL"] = std::vector<std::string>{"true"};
        parameterMap["MovingImagePyramid"] = std::vector<std::string>{"OpenCLMovingGenericImagePyramid"};
        parameterMap["OpenCLMovingGenericImagePyramidUseOpenCL"] = std::vector<std::string>{"true"};
        parameterMap["Resampler"] = std::vector<std::string>{"OpenCLResampler"};
        parameterMap["OpenCLResamplerUseOpenCL"] = std::vector<std::string>{"true"};
    }

    parameterObject->SetParameterMap(parameterMap);
    parameterObject->Print(std::cout);
    elastix->SetParameterObject(parameterObject);

    auto start = std::chrono::high_resolution_clock::now();

    try
    {
        elastix->Update();
    }
    catch (itk::ExceptionObject & error)
    {
        std::cerr << "Error: " << error << std::endl;
        return;
    }
    auto result = elastix->GetOutput();

    std::cout << "Registration completed successfully!" << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Registration completed in " << duration.count() << " seconds." << std::endl;


    using WriterType = itk::ImageFileWriter<ImageType>;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName("elastix_output/result.0.nii");
    writer->SetInput(result);

    try
    {
        writer->Update();
    }
    catch (itk::ExceptionObject & error)
    {
        std::cerr << "Error: " << error << std::endl;
    }
}

void performRegistration(ImageType::Pointer fixedImage, ImageType::Pointer movingImage, bool useOpenCL, bool useElxLib)
{
    // Use elastix executable
    if (!useElxLib)
    {
        // Execute elastix
        int ret = executeElastix("cbct_image.nii", "ct_image.nii", "elastix_output", "elastix_parameters.txt");
        if (ret != 0)
            std::cerr << "Elastix execution failed with error code: " << ret << std::endl;
        std::cout << "Elastix registration completed successfully." << std::endl;
    }
    else{
        libElastix(fixedImage, movingImage, useOpenCL);
    }
    return;
}

int main(int argc, char * argv[])
{
    auto mvimg = load_ct();
    auto fximg = load_cbct();

    saveSliceToPNG<ImageType>(mvimg, 256, "mvimg.png"); 
    saveSliceToPNG<ImageType>(fximg, 256, "fximg.png");

    bool useOpenCL = true;  // Set to true to use gpu
    bool useElxLib = false; // Set to true to use elastix library or false to use elastix executable
    performRegistration(fximg, mvimg, useOpenCL, useElxLib);

    // Load the result image
    auto result = loadITKImage("elastix_output/result.0.nii");
    saveSliceToPNG<ImageType>(result, 256, "result.png");

    return EXIT_SUCCESS;
}