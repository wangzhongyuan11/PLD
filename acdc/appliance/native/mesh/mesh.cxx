// Board-side, headless surface generation. Labels/statistics are never modified.
#include <vtkCellArray.h>
#include <vtkDataArray.h>
#include <vtkDiscreteMarchingCubes.h>
#include <vtkImageData.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkPLYWriter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkSMPTools.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkWindowedSincPolyDataFilter.h>
#include <vtkXMLPolyDataWriter.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock=std::chrono::steady_clock;

static void write_mesh(vtkPolyData* source,const fs::path& prefix,
                       const std::array<unsigned char,3>& color) {
    vtkNew<vtkPolyDataNormals> normals;
    normals->SetInputData(source);normals->SplittingOff();normals->ConsistencyOn();normals->Update();
    auto mesh=normals->GetOutput();
    vtkNew<vtkXMLPolyDataWriter> xml;
    xml->SetFileName((prefix.string()+".vtp").c_str());xml->SetInputData(mesh);
    if(xml->Write()!=1) throw std::runtime_error("VTP write failed");
    if(mesh->GetNumberOfPoints()) {
        vtkNew<vtkPLYWriter> ply;
        ply->SetFileName((prefix.string()+".ply").c_str());ply->SetInputData(mesh);
        ply->SetFileTypeToBinary();ply->SetColorModeToUniformColor();ply->SetColor(color[0],color[1],color[2]);
        if(ply->Write()!=1) throw std::runtime_error("PLY write failed");
    } else {
        std::ofstream f(prefix.string()+".ply");
        f << "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
             "element face 0\nproperty list uchar int vertex_indices\nend_header\n";
        if(!f) throw std::runtime_error("empty PLY write failed");
    }
    if(mesh->GetNumberOfPoints()>std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("mesh index overflow");
    std::ofstream points(prefix.string()+".points.f32",std::ios::binary);
    std::ofstream directions(prefix.string()+".normals.f32",std::ios::binary);
    std::ofstream indices(prefix.string()+".indices.u32",std::ios::binary);
    auto n=mesh->GetPointData()->GetNormals();
    for(vtkIdType i=0;i<mesh->GetNumberOfPoints();++i) {
        double p[3],normal[3];mesh->GetPoint(i,p);
        if(!n) throw std::runtime_error("missing normals");
        n->GetTuple(i,normal);
        for(int a=0;a<3;++a) {
            float x=static_cast<float>(p[a]),y=static_cast<float>(normal[a]);
            points.write(reinterpret_cast<const char*>(&x),sizeof(x));
            directions.write(reinterpret_cast<const char*>(&y),sizeof(y));
        }
    }
    auto cells=mesh->GetPolys();
    cells->InitTraversal();vtkIdType count;const vtkIdType* ids;
    while(cells->GetNextCell(count,ids)) {
        if(count!=3) throw std::runtime_error("non-triangle mesh");
        for(int a=0;a<3;++a) {
            uint32_t id=static_cast<uint32_t>(ids[a]);
            indices.write(reinterpret_cast<const char*>(&id),sizeof(id));
        }
    }
    points.close();directions.close();indices.close();
    if(!points || !directions || !indices) throw std::runtime_error("binary mesh write failed");
    std::ofstream metadata(prefix.string()+".json");
    metadata << "{\"points\":" << mesh->GetNumberOfPoints()
             << ",\"triangles\":" << mesh->GetNumberOfPolys()
             << ",\"generator\":\"pynq_ps\",\"quantitative\":false,\"coordinate_system\":\"RAS\",\"units\":\"mm\"}\n";
    if(!metadata) throw std::runtime_error("mesh metadata write failed");
}

int main(int argc,char** argv) {
    try {
        if(argc!=7) throw std::runtime_error("usage: acdc_mesh labels.u8 nx ny nz affine.txt new_output_directory");
        const uint16_t one=1;
        if(*reinterpret_cast<const unsigned char*>(&one)!=1) throw std::runtime_error("little endian required");
        std::array<int,3> shape={std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4])};
        uint64_t size=1;
        for(int n:shape) {if(n<1 || n>4096) throw std::runtime_error("invalid shape");size*=n;}
        if(size>2000000) throw std::runtime_error("mesh input limit exceeded");
        if(fs::file_size(argv[1])!=size) throw std::runtime_error("label byte length mismatch");
        std::vector<unsigned char> labels(size);
        std::ifstream input(argv[1],std::ios::binary);
        input.read(reinterpret_cast<char*>(labels.data()),size);
        if(!input || std::any_of(labels.begin(),labels.end(),[](unsigned char c){return c>3;}))
            throw std::runtime_error("invalid label data");
        vtkNew<vtkMatrix4x4> matrix;
        std::ifstream affine(argv[5]);
        for(int i=0;i<4;++i) for(int j=0;j<4;++j) {
            double value;
            if(!(affine>>value) || !std::isfinite(value)) throw std::runtime_error("invalid affine");
            matrix->SetElement(i,j,value);
        }
        fs::path directory(argv[6]);
        if(fs::exists(directory)) throw std::runtime_error("output directory must not exist");
        fs::create_directories(directory);
        vtkSMPTools::Initialize(1);
        const std::array<std::string,4> names={"background","RV","MYO","LV"};
        const std::array<std::array<unsigned char,3>,4> colors={{{0,0,0},{255,0,0},{0,255,0},{0,0,255}}};
        for(int c=1;c<4;++c) {
            auto start=Clock::now();
            std::array<int,3> low=shape,high={-1,-1,-1};bool clipped=false;
            for(int x=0;x<shape[0];++x) for(int y=0;y<shape[1];++y) for(int z=0;z<shape[2];++z) {
                if(labels[(static_cast<size_t>(x)*shape[1]+y)*shape[2]+z]!=c) continue;
                const std::array<int,3> p={x,y,z};
                for(int a=0;a<3;++a) {low[a]=std::min(low[a],p[a]);high[a]=std::max(high[a],p[a]);
                    clipped=clipped || p[a]==0 || p[a]==shape[a]-1;}
            }
            vtkNew<vtkPolyData> raw;
            if(high[0]>=0) {
                std::array<int,3> padded;
                for(int a=0;a<3;++a) padded[a]=high[a]-low[a]+3;
                vtkNew<vtkImageData> image;image->SetDimensions(padded.data());
                image->SetOrigin(low[0]-1,low[1]-1,low[2]-1);image->AllocateScalars(VTK_UNSIGNED_CHAR,1);
                auto data=static_cast<unsigned char*>(image->GetScalarPointer());
                std::fill(data,data+static_cast<size_t>(padded[0])*padded[1]*padded[2],0);
                for(int x=low[0];x<=high[0];++x) for(int y=low[1];y<=high[1];++y) for(int z=low[2];z<=high[2];++z)
                    data[(static_cast<size_t>(z-low[2]+1)*padded[1]+y-low[1]+1)*padded[0]+x-low[0]+1]=
                        labels[(static_cast<size_t>(x)*shape[1]+y)*shape[2]+z]==c;
                vtkNew<vtkDiscreteMarchingCubes> contour;contour->SetInputData(image);
                contour->SetValue(0,1);contour->ComputeNormalsOff();
                vtkNew<vtkTransform> transform;transform->SetMatrix(matrix);
                vtkNew<vtkTransformPolyDataFilter> physical;
                physical->SetInputConnection(contour->GetOutputPort());physical->SetTransform(transform);physical->Update();
                raw->DeepCopy(physical->GetOutput());
            }
            write_mesh(raw,directory/(names[c]+".raw"),colors[c]);
            vtkNew<vtkPolyData> display;
            if(raw->GetNumberOfPoints()) {
                vtkNew<vtkLinearSubdivisionFilter> subdivide;subdivide->SetInputData(raw);
                subdivide->SetNumberOfSubdivisions(1);
                vtkNew<vtkWindowedSincPolyDataFilter> relax;
                relax->SetInputConnection(subdivide->GetOutputPort());relax->SetNumberOfIterations(80);
                relax->SetPassBand(.005);relax->SetWindowFunctionToNuttall();
                relax->BoundarySmoothingOff();relax->FeatureEdgeSmoothingOff();relax->NonManifoldSmoothingOff();
                relax->NormalizeCoordinatesOn();relax->Update();display->DeepCopy(relax->GetOutput());
            }
            write_mesh(display,directory/(names[c]+".smooth"),colors[c]);
            double seconds=std::chrono::duration<double>(Clock::now()-start).count();
            std::ofstream report(directory/(names[c]+".json"));
            report << "{\"label\":" << c << ",\"fov_truncated\":" << (clipped?"true":"false")
                   << ",\"seconds\":" << seconds << ",\"subdivisions\":1,\"iterations\":80,\"pass_band\":0.005,\"window\":\"nuttall\"}\n";
            if(!report) throw std::runtime_error("surface report write failed");
        }
        return 0;
    } catch(const std::exception& error) {std::cerr << error.what() << '\n';return 1;}
}
