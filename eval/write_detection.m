clear;
seq_name = 'my_measure';
model_name = 'CPM';
target_dir = fullfile(seq_name, model_name, 'mat');
if ~exist(target_dir, 'dir')
    mkdir(target_dir)
end

filetext = fileread('detection_2d.json');
json = jsondecode(filetext);

for i=1:length(json)
    json_data = json(i);
    img_name = json_data.img_dir;
    basename = strsplit(img_name, '/');
    basename = basename{end};
    basename = strsplit(basename, '.');
    basename = {basename{1:end-1}};
    basename = join(basename, '.');
    basename = basename{1};

    file_in = strcat('/media/posefs3b/Users/tsimon/outputs/hands/hand_mpi_nzl_v20_orig_ts/hands_v143_noft_102k/mat/', basename, '.mat.mat');
    data = load(file_in);
    
    hand2d = json_data.hand2d;
    hand2d = [hand2d, ones(size(hand2d, 1), 1)];

    vertices = reshape(hand2d, 1, 63);

    if ~isempty(data.left_hand)
        data.left_hand.vertices = vertices;
        data.left_hand.prediction = hand2d;
    end

    if ~isempty(data.right_hand)
        data.right_hand.vertices = vertices;
        data.right_hand.prediction = hand2d;
    end

    file_out = fullfile(target_dir, strcat(basename, '.mat.mat'));
    save(file_out, '-struct', 'data');

end

