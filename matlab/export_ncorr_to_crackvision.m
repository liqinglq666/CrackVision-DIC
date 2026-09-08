function export_ncorr_to_crackvision(source, output_file, sampling_interval_s, precision, start_time_s)
%EXPORT_NCORR_TO_CRACKVISION Export only the Ncorr fields needed by CrackVision-DIC.
%
% New analysis (recommended):
%   handles_ncorr = ncorr;
%   ... finish Format Displacements, unit conversion in mm, Calculate Strains ...
%   export_ncorr_to_crackvision(handles_ncorr,'Specimen01_CrackVision.h5',5)
%
% Existing saved Ncorr MAT:
%   export_ncorr_to_crackvision('Specimen01.mat','Specimen01_CrackVision.h5',5)
%
% The bridge keeps reference-formatted U/V and Green-Lagrange Exx/Eyy/Exy,
% finite-data mask, time and physical scale metadata. It deliberately omits
% images, ROI objects, correlation plots and unrelated Ncorr state.

    if nargin < 3 || isempty(sampling_interval_s)
        sampling_interval_s = 1;
    end
    if nargin < 4 || isempty(precision)
        precision = 'single';
    end
    if nargin < 5 || isempty(start_time_s)
        start_time_s = 0;
    end

    if ~isscalar(sampling_interval_s) || ~isfinite(sampling_interval_s) || sampling_interval_s <= 0
        error('sampling_interval_s must be a finite positive scalar.');
    end
    if ~isscalar(start_time_s) || ~isfinite(start_time_s)
        error('start_time_s must be finite.');
    end

    precision = lower(char(precision));
    if ~ismember(precision, {'single','double'})
        error('precision must be ''single'' or ''double''.');
    end

    output_file = char(output_file);
    [~,~,ext] = fileparts(output_file);
    if ~ismember(lower(ext), {'.h5','.hdf5'})
        error('output_file must end with .h5 or .hdf5.');
    end

    data_dic = resolve_ncorr_data(source);
    if ~isfield(data_dic,'displacements') || isempty(data_dic.displacements)
        error('Ncorr displacement results are missing.');
    end
    if ~isfield(data_dic,'strains') || isempty(data_dic.strains)
        error('Ncorr strain results are missing. Run Analysis -> Calculate Strains.');
    end
    if ~isfield(data_dic,'dispinfo') || isempty(data_dic.dispinfo)
        error('Ncorr dispinfo metadata is missing.');
    end

    n_frames = numel(data_dic.displacements);
    if n_frames ~= numel(data_dic.strains) || n_frames < 1
        error('Ncorr displacement/strain frame count is invalid.');
    end

    dispinfo = data_dic.dispinfo(1);
    pixel_size_mm = required_scalar(dispinfo,'pixtounits');
    spacing_raw = required_scalar(dispinfo,'spacing');
    if spacing_raw < 0
        error('Ncorr spacing must be >= 0.');
    end

    units = '';
    if isfield(dispinfo,'units') && ~isempty(dispinfo.units)
        units = strtrim(char(dispinfo.units));
    end
    if isempty(units)
        warning('dispinfo.units is empty; pixtounits is assumed to be mm/pixel.');
        units = 'mm_assumed';
    elseif ~is_mm_unit(units)
        error(['Ncorr units are "%s", not mm. Re-run Format Displacements -> ' ...
               'Get Unit Conversion using a millimetre scale.'], units);
    end

    % Native Ncorr spacing is a skipped-pixel count.
    dic_step_px = spacing_raw + 1;
    dic_point_spacing_mm = pixel_size_mm * dic_step_px;

    [u0,v0,exx0,eyy0,exy0] = read_reference_frame(data_dic,1);
    validate_frame(1,u0,v0,exx0,eyy0,exy0);
    [height,width] = size(u0);

    if exist(output_file,'file')
        delete(output_file);
    end

    % MATLAB/HDF5 dimension order is reversed as seen by h5py. Creating
    % [width,height,frame] and writing transposed 2D slabs makes Python see
    % [frame,height,width] with the original image orientation.
    dims = [width,height,n_frames];
    chunks = [min(width,128),min(height,128),1];
    create_float_dataset(output_file,'/fields/u',dims,chunks,precision);
    create_float_dataset(output_file,'/fields/v',dims,chunks,precision);
    create_float_dataset(output_file,'/fields/exx',dims,chunks,precision);
    create_float_dataset(output_file,'/fields/eyy',dims,chunks,precision);
    create_float_dataset(output_file,'/fields/exy',dims,chunks,precision);
    h5create(output_file,'/fields/mask',dims,'Datatype','uint8', ...
        'ChunkSize',chunks,'Deflate',4);

    h5create(output_file,'/time_s',[1,n_frames],'Datatype','double');
    time_s = start_time_s + (0:n_frames-1) * sampling_interval_s;
    h5write(output_file,'/time_s',time_s);

    for i = 1:n_frames
        if i == 1
            u=u0; v=v0; exx=exx0; eyy=eyy0; exy=exy0;
        else
            [u,v,exx,eyy,exy] = read_reference_frame(data_dic,i);
            validate_frame(i,u,v,exx,eyy,exy);
            if ~isequal(size(u),[height,width])
                error('Frame %d matrix size changed.',i);
            end
        end

        mask = isfinite(u) & isfinite(v) & isfinite(exx) & isfinite(eyy) & isfinite(exy);
        start = [1,1,i];
        count = [width,height,1];
        h5write(output_file,'/fields/u',reshape(cast(u.',precision),[width,height,1]),start,count);
        h5write(output_file,'/fields/v',reshape(cast(v.',precision),[width,height,1]),start,count);
        h5write(output_file,'/fields/exx',reshape(cast(exx.',precision),[width,height,1]),start,count);
        h5write(output_file,'/fields/eyy',reshape(cast(eyy.',precision),[width,height,1]),start,count);
        h5write(output_file,'/fields/exy',reshape(cast(exy.',precision),[width,height,1]),start,count);
        h5write(output_file,'/fields/mask',reshape(uint8(mask.'),[width,height,1]),start,count);

        if mod(i,max(1,floor(n_frames/20))) == 0 || i == n_frames
            fprintf('CrackVision export: %d / %d frames\n',i,n_frames);
        end
    end

    h5writeatt(output_file,'/','format','CrackVision-Ncorr');
    h5writeatt(output_file,'/','format_version',int32(1));
    h5writeatt(output_file,'/','producer','export_ncorr_to_crackvision.m');
    h5writeatt(output_file,'/','coordinate_system','reference');
    h5writeatt(output_file,'/','displacement_fields','plot_u_ref_formatted,plot_v_ref_formatted');
    h5writeatt(output_file,'/','strain_measure','Green-Lagrange');
    h5writeatt(output_file,'/','strain_fields','plot_exx_ref_formatted,plot_eyy_ref_formatted,plot_exy_ref_formatted');
    h5writeatt(output_file,'/','pixel_size_mm',double(pixel_size_mm));
    h5writeatt(output_file,'/','ncorr_spacing_raw',double(spacing_raw));
    h5writeatt(output_file,'/','dic_step_px',double(dic_step_px));
    h5writeatt(output_file,'/','dic_point_spacing_mm',double(dic_point_spacing_mm));
    h5writeatt(output_file,'/','sampling_interval_s',double(sampling_interval_s));
    h5writeatt(output_file,'/','start_time_s',double(start_time_s));
    h5writeatt(output_file,'/','numeric_precision',precision);
    h5writeatt(output_file,'/','source_units',units);
    h5writeatt(output_file,'/','frame_count',int32(n_frames));
    h5writeatt(output_file,'/','height',int32(height));
    h5writeatt(output_file,'/','width',int32(width));

    info = dir(output_file);
    fprintf('\nCrackVision-Ncorr bridge created.\n');
    fprintf('File: %s\n',output_file);
    fprintf('Frames: %d | DIC matrix: %d x %d\n',n_frames,height,width);
    fprintf('Scale: %.9g mm/px | spacing: %.9g -> step %.9g px\n', ...
        pixel_size_mm,spacing_raw,dic_step_px);
    fprintf('Grid spacing: %.9g mm/point | precision: %s\n', ...
        dic_point_spacing_mm,precision);
    fprintf('File size: %.2f MB\n',info.bytes/1024/1024);
end


function data_dic = resolve_ncorr_data(source)
    if ischar(source) || (isstring(source) && isscalar(source))
        source_file = char(source);
        if ~exist(source_file,'file')
            error('Ncorr MAT file does not exist: %s',source_file);
        end
        vars = whos('-file',source_file);
        if ~ismember('data_dic_save',{vars.name})
            error('Saved MAT does not contain data_dic_save.');
        end
        fprintf('Loading data_dic_save from %s ...\n',source_file);
        loaded = load(source_file,'data_dic_save');
        data_dic = loaded.data_dic_save;
        return;
    end

    if isobject(source) && isprop(source,'data_dic')
        data_dic = source.data_dic;
        return;
    end

    if isstruct(source)
        if isfield(source,'data_dic_save')
            data_dic = source.data_dic_save;
        elseif isfield(source,'displacements') && isfield(source,'strains')
            data_dic = source;
        else
            error('Struct source is not Ncorr data_dic.');
        end
        return;
    end

    error('Pass a live Ncorr handle, Ncorr data_dic struct, or saved Ncorr MAT path.');
end


function [u,v,exx,eyy,exy] = read_reference_frame(data_dic,index)
    d = frame_item(data_dic.displacements,index);
    s = frame_item(data_dic.strains,index);
    u = required_field(d,'plot_u_ref_formatted',index);
    v = required_field(d,'plot_v_ref_formatted',index);
    exx = required_field(s,'plot_exx_ref_formatted',index);
    eyy = required_field(s,'plot_eyy_ref_formatted',index);
    exy = required_field(s,'plot_exy_ref_formatted',index);
    u=double(u); v=double(v); exx=double(exx); eyy=double(eyy); exy=double(exy);
end


function item = frame_item(container,index)
    if iscell(container)
        item = container{index};
    else
        item = container(index);
    end
end


function value = required_field(item,name,frame_index)
    if ~isstruct(item) || ~isfield(item,name) || isempty(item.(name))
        error(['Frame %d is missing %s. Finish Ncorr Analysis -> Format ' ...
               'Displacements and Analysis -> Calculate Strains first.'], ...
              frame_index,name);
    end
    value = item.(name);
    if ~isnumeric(value)
        error('Frame %d field %s is not numeric.',frame_index,name);
    end
end


function value = required_scalar(item,name)
    if ~isstruct(item) || ~isfield(item,name) || isempty(item.(name))
        error('Ncorr dispinfo.%s is missing.',name);
    end
    value = double(item.(name));
    if ~isscalar(value) || ~isfinite(value)
        error('Ncorr dispinfo.%s must be a finite scalar.',name);
    end
end


function validate_frame(index,u,v,exx,eyy,exy)
    arrays = {u,v,exx,eyy,exy};
    target = size(u);
    for k = 1:numel(arrays)
        if ~ismatrix(arrays{k}) || ~isequal(size(arrays{k}),target)
            error('Frame %d U/V/Exx/Eyy/Exy matrices do not share one size.',index);
        end
    end
end


function create_float_dataset(file,path,dims,chunks,precision)
    h5create(file,path,dims,'Datatype',precision,'ChunkSize',chunks,'Deflate',4);
end


function tf = is_mm_unit(text)
    normalized = lower(strtrim(char(text)));
    tf = ismember(normalized,{'mm','millimeter','millimeters','millimetre','millimetres'});
end
