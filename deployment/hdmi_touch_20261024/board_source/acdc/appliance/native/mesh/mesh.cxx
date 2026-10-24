// Board-side, headless surface generation. Labels/statistics are never modified.
#include <vtkCellArray.h>
#include <vtkDataArray.h>
#include <vtkDiscreteMarchingCubes.h>
#include <vtkImageData.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkSMPTools.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkWindowedSincPolyDataFilter.h>
#include <vtkVersion.h>
#include <vtkXMLPolyDataWriter.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock=std::chrono::steady_clock;

static std::array<double,4> write_mesh(vtkPolyData* source,const fs::path& prefix,
                       const std::array<unsigned char,3>& color) {
    auto start=Clock::now();
    vtkNew<vtkPolyDataNormals> normals;
    normals->SetInputData(source);normals->SplittingOff();normals->ConsistencyOn();normals->Update();
    auto mesh=normals->GetOutput();
    double normals_seconds=std::chrono::duration<double>(Clock::now()-start).count();start=Clock::now();
    vtkNew<vtkXMLPolyDataWriter> xml;
    xml->SetFileName((prefix.string()+".vtp").c_str());xml->SetInputData(mesh);
    // The final ZIP performs one level-1 compression pass. Avoid zlib/base64
    // here: it dominated the board's geometry export time and then ran again.
    xml->SetCompressorTypeToNone();xml->SetDataModeToAppended();xml->EncodeAppendedDataOff();
    if(xml->Write()!=1) throw std::runtime_error("VTP write failed");
    double vtp_seconds=std::chrono::duration<double>(Clock::now()-start).count();start=Clock::now();
    if(mesh->GetNumberOfPoints()>std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("mesh index overflow");
    size_t vertices=static_cast<size_t>(mesh->GetNumberOfPoints()),triangles=static_cast<size_t>(mesh->GetNumberOfPolys());
    std::vector<float> positions(vertices*3),normals_data(vertices*3);
    std::vector<uint32_t> connectivity(triangles*3);
    std::vector<unsigned char> ply_vertices(vertices*27),ply_faces(triangles*16);
    auto n=mesh->GetPointData()->GetNormals();
    for(vtkIdType i=0;i<mesh->GetNumberOfPoints();++i) {
        double p[3],normal[3];mesh->GetPoint(i,p);
        if(!n) throw std::runtime_error("missing normals");
        n->GetTuple(i,normal);
        for(int a=0;a<3;++a) {
            float x=static_cast<float>(p[a]),y=static_cast<float>(normal[a]);
            positions[i*3+a]=x;normals_data[i*3+a]=y;
            std::memcpy(ply_vertices.data()+i*27+a*4,&x,4);
            std::memcpy(ply_vertices.data()+i*27+12+a*4,&y,4);
        }
        std::copy(color.begin(),color.end(),ply_vertices.data()+i*27+24);
    }
    auto cells=mesh->GetPolys();
    cells->InitTraversal();vtkIdType count;const vtkIdType* ids;size_t face=0;
    while(cells->GetNextCell(count,ids)) {
        if(count!=3) throw std::runtime_error("non-triangle mesh");
        ply_faces[face*16]=3;
        for(int a=0;a<3;++a) {
            uint32_t id=static_cast<uint32_t>(ids[a]);
            connectivity[face*3+a]=id;std::memcpy(ply_faces.data()+face*16+1+a*4,&id,4);
        }
        std::copy(color.begin(),color.end(),ply_faces.data()+face*16+13);++face;
    }
    if(face!=triangles) throw std::runtime_error("mesh triangle count mismatch");
    // Bulk binary PLY retains coordinates, normals, colors and triangles while
    // avoiding millions of tiny formatted/stream writes on Cortex-A9.
    std::ofstream ply(prefix.string()+".ply",std::ios::binary);
    ply << "ply\nformat binary_little_endian 1.0\ncomment ACDC board surface\nelement vertex " << vertices
        << "\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nelement face " << triangles
        << "\nproperty list uchar int vertex_indices\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    if(!ply_vertices.empty()) ply.write(reinterpret_cast<const char*>(ply_vertices.data()),ply_vertices.size());
    if(!ply_faces.empty()) ply.write(reinterpret_cast<const char*>(ply_faces.data()),ply_faces.size());
    ply.close();if(!ply) throw std::runtime_error("PLY write failed");
    double ply_seconds=std::chrono::duration<double>(Clock::now()-start).count();start=Clock::now();
    auto binary=[&](const std::string& suffix,const void* bytes,size_t length) {
        std::ofstream stream(prefix.string()+suffix,std::ios::binary);
        if(length) stream.write(static_cast<const char*>(bytes),length);
        stream.close();if(!stream) throw std::runtime_error("binary mesh write failed");
    };
    binary(".points.f32",positions.data(),positions.size()*sizeof(float));
    binary(".normals.f32",normals_data.data(),normals_data.size()*sizeof(float));
    binary(".indices.u32",connectivity.data(),connectivity.size()*sizeof(uint32_t));
    std::ofstream metadata(prefix.string()+".json");
    metadata << "{\"points\":" << mesh->GetNumberOfPoints()
             << ",\"triangles\":" << mesh->GetNumberOfPolys()
             << ",\"generator\":\"pynq_ps\",\"quantitative\":false,\"coordinate_system\":\"RAS\",\"units\":\"mm\"}\n";
    if(!metadata) throw std::runtime_error("mesh metadata write failed");
    return {normals_seconds,vtp_seconds,ply_seconds,std::chrono::duration<double>(Clock::now()-start).count()};
}

int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--version") {
            std::cout << "{\"abi\":1,\"vtk\":\"" << vtkVersion::GetVTKVersion()
                      << "\",\"pointer_bits\":" << 8*sizeof(void*) << "}\n";
            return 0;
        }
        if(argc!=7 && argc!=10) throw std::runtime_error("usage: acdc_mesh labels.u8 nx ny nz affine.txt new_output_directory [subdivisions iterations pass_band]");
        int subdivisions=argc==10?std::stoi(argv[7]):1,iterations=argc==10?std::stoi(argv[8]):80;
        double pass_band=argc==10?std::stod(argv[9]):.005;
        if(subdivisions<0 || subdivisions>1 || iterations<1 || iterations>100 || !std::isfinite(pass_band) || pass_band<=0. || pass_band>=2.)
            throw std::runtime_error("invalid surface filter parameters");
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
        // Meshing runs after inference; both Cortex-A9 cores are available.
        vtkSMPTools::Initialize(2);
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
            double extraction_seconds=std::chrono::duration<double>(Clock::now()-start).count();
            auto raw_times=write_mesh(raw,directory/(names[c]+".raw"),colors[c]);
            auto filter_start=Clock::now();
            vtkNew<vtkPolyData> display;
            if(raw->GetNumberOfPoints()) {
                vtkNew<vtkLinearSubdivisionFilter> subdivide;subdivide->SetInputData(raw);
                subdivide->SetNumberOfSubdivisions(subdivisions);
                vtkNew<vtkWindowedSincPolyDataFilter> relax;
                relax->SetInputConnection(subdivide->GetOutputPort());relax->SetNumberOfIterations(iterations);
                relax->SetPassBand(pass_band);relax->SetWindowFunctionToNuttall();
                relax->BoundarySmoothingOff();relax->FeatureEdgeSmoothingOff();relax->NonManifoldSmoothingOff();
                relax->NormalizeCoordinatesOn();relax->Update();display->DeepCopy(relax->GetOutput());
            }
            double smoothing_seconds=std::chrono::duration<double>(Clock::now()-filter_start).count();
            auto display_times=write_mesh(display,directory/(names[c]+".smooth"),colors[c]);
            double seconds=std::chrono::duration<double>(Clock::now()-start).count();
            std::ofstream report(directory/(names[c]+".json"));
            report << "{\"label\":" << c << ",\"fov_truncated\":" << (clipped?"true":"false")
                   << ",\"seconds\":" << seconds << ",\"extraction_seconds\":" << extraction_seconds
                   << ",\"smoothing_seconds\":" << smoothing_seconds << ",\"raw_stages\":[" << raw_times[0] << ',' << raw_times[1] << ',' << raw_times[2] << ',' << raw_times[3]
                   << "],\"smooth_stages\":[" << display_times[0] << ',' << display_times[1] << ',' << display_times[2] << ',' << display_times[3]
                   << "],\"stage_order\":[\"normals\",\"vtp\",\"ply\",\"index\"],\"subdivisions\":" << subdivisions
                   << ",\"iterations\":" << iterations << ",\"pass_band\":" << pass_band << ",\"window\":\"nuttall\"}\n";
            if(!report) throw std::runtime_error("surface report write failed");
        }
        return 0;
    } catch(const std::exception& error) {std::cerr << error.what() << '\n';return 1;}
}
