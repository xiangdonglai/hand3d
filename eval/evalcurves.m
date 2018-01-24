% Demo to show simple examples to use Panoptic Dataset
seq_name = 'hand_mpi_nzl_v20_orig_ts';

% Data is kept in json format
addpath(genpath('/media/posefs1b/Users/tsimon/loop/devel/panoptic-toolbox/matlab'));
addpath('/media/posefs1b/Users/tsimon/loop/devel/Convolutional-Pose-Machines/testing/src/');
drange = linspace(0,1,250);
%     hand_iteration_model = '/model/hands_v141_han_iter1/pose_iter_340000.caffemodel';
% models = {'hands_v12_reno_noscale_271',
%     'hands_v12_reno_noscale_180k',
%     'hands_v121_han_iter1',
%     'hands_v123_han_iter2_ft0_bb',
%     'hands_v13_pure_180k',
%     'hands_v13_pure_100k',
%     'hands_v13_pure_120k',
%     'hands_v131_han_iter1',
%     'hands_v133_han_iter2_ft0_bb',
%     'hands_v133_han_iter2_noft_180k',
%     'hands_v14_mix',
%     'hands_v14_mix_180k',
%     'hands_v141_han_iter1',
%     'hands_v142_han_iter2_ft0',
%     'hands_v143_han_iter2_ft0_340k',
%     'hands_v143_han_iter2_noft_174k'};


%     'hands_v12_reno_noscale_30k',
%     'hands_v123_noft_80k',
%     'hands_v123_noft_120k',
%     'hands_v13_pure_30k',
%     'hands_v133_noft_120k',
%     'hands_v133_han_iter2_noft_180k',
models = {
    'hands_v12_reno_noscale_32k',
    'hands_v123_noft_120k',
    'hands_v13_pure_30k',
    'hands_v133_noft_122k',
    'hands_v14_mix_50k',
    'hands_v143_noft_102k',
    'CPM'};

close all;
clear results;
do_plot = 0;
lentest = nan;
for idm = 1:length(models)
    model_name = models{idm};
    % clear hand_iteration_model;
    
    test_files = {};
    warning off;
    if idm <= 6
        in_path = sprintf('/media/posefs3b/Users/tsimon/outputs/hands/%s/%s/mat/', seq_name, model_name);
    else
        in_path = sprintf('./%s/%s/mat/', seq_name, model_name);
    end
    % for iddirs=1:length(test_dirs)
    Dfiles = dir( fullfile(in_path, '*.mat_l.json') );
    if ~isempty(Dfiles)
        for idf=1:length(Dfiles)
            fname = Dfiles(idf).name;
            fname_out = fname(1:end-7);
            movefile(fullfile( in_path, fname), fullfile(in_path,fname_out));
        end
    end
    Dfiles = dir( fullfile(in_path, '*.mat') );
    for idf=1:length(Dfiles)
        test_files = {test_files{:}, fullfile(in_path, Dfiles(idf).name) };
    end
    % end
    
    %
    if idm==1
        lentest=length(test_files);
    else
        assert(lentest==length(test_files));
    end

    all_dist = zeros(21, length(test_files));
    all_types = zeros(length(test_files),1);
    for idf=1:length(test_files)
        hentry = load(test_files{idf});
        test_imagen = sprintf('%s', hentry.originim);
        test_image = test_imagen;
        
        if strcmp(hentry.ver, 'lwt')
            warning('This doesnt work on non-mpi');
        else
        % Added for MPI person finding

            [~,nm,~] = fileparts(test_files{idf});
            A = sscanf(nm, '%d_%d');
            if length(A)~=2
                error('invalid name format for mpi');
            end

            id_image = A(1);
            id_person = A(2);
            A = sscanf(hentry.annot.image.name, '%d');
            assert(id_image==A);
        
        end
        if hentry.is_left
            prediction = hentry.left_hand.prediction;
        else
            prediction = hentry.right_hand.prediction;
        end
        
        
        if do_plot
            if ~ishandle(1);
                figure(1);
            else
                set(0, 'CurrentFigure', 1);
            end
            im = imread(test_image);
            hold off;
            imshow(im);
            hold on;
            plot( hentry.hand_pts(:,1), hentry.hand_pts(:,2), 'g.');
            plot( prediction(:,1), prediction(:,2), 'bo');
            plot([hentry.annot.annorect(id_person).annopoints.point.x], [hentry.annot.annorect(id_person).annopoints.point.y], 'c*');
            drawnow;
            pause;
        end
        
        dist = sqrt(sum((hentry.hand_pts(:,1:2)-prediction(:,1:2)).^2,2));
        dist = dist/(hentry.bbsqsize);
        all_dist(:, idf) = dist;
        all_types(idf) = 1;
        if strcmp(hentry.ver, 'lwt')
            all_types(idf) = 2;
        end
    end
    
    pck = zeros(numel(drange),21);
    for jidx = 1:21
        % compute PCK for each threshold
        for k = 1:numel(drange)
            d = squeeze(all_dist(jidx,:));
            % dist is NaN if gt is missing; ignore dist in this case
            pck(k,jidx) = pck(k,jidx) + mean(d(~isnan(d))<=drange(k));
        end
    end
    result.all_dist = all_dist;
    result.all_types = all_types;
    result.pck = pck;
    result.model_name = model_name;
    results(idm) = result;
end
%%
drange = linspace(0,1,250);
for idr=1:length(results)
    result = results(idr);
    pck = zeros(numel(drange),21);
    for jidx = 1:21
        % compute PCK for each threshold
        for k = 1:numel(drange)
%             sel_files = result.all_types==1;
            sel_files = result.all_types~=-1;
            d = squeeze(result.all_dist(jidx,sel_files));
            % dist is NaN if gt is missing; ignore dist in this case
            pck(k,jidx) = pck(k,jidx) + mean(d(~isnan(d))<=drange(k));
        end
    end
%     result.pck = pck;
%     results(idr) = result;
    figure(100); 
    plot(drange, mean(pck(:,:),2), 'LineWidth', 2); axis([0 1 0 1]); grid on; hold all;
end
legend(models);

%%
figure(101);
clf;
set(gcf, 'Color', 'w');
% set(gcf, 'Position', [0 0 400 400]);

drange = linspace(0,1,250);
% sel_results = [2, 4, 5, 8, 10, 14];
sel_results = 1:7;
texts = {'Render (initial)', 'Render (iteration 2)', 'Manual (initial)', 'Manual (iteration 2)', 'Mix (initial)', 'Mix (iteration 3)', 'CPM'};
styles = {':', '-', ':', '-', ':', '-', ':','-', ':'}; 
cols =[    0.8941    0.1020    0.1098
    0.3020    0.6863    0.2902
    0.2157    0.4941    0.7216
    ];
cols = kron(cols, [0.85; 1]);
for idri=1:length(sel_results)
    idr =sel_results(idri);
    result = results(idr);
    pck = zeros(numel(drange),21);
    for jidx = 1:21
        % compute PCK for each threshold
        for k = 1:numel(drange)
%             d = squeeze(result.all_dist(jidx,:));
            sel_files = result.all_types==2;
%             sel_files = result.all_types~=-1;
            d = squeeze(result.all_dist(jidx,sel_files));

            % dist is NaN if gt is missing; ignore dist in this case
            pck(k,jidx) = pck(k,jidx) + mean(d(~isnan(d))<=drange(k));
        end
    end
%     result.pck = pck;
%     results(idr) = result; 
    % plot(drange, mean(pck(:,:),2), 'Color', cols(idri, :), 'LineWidth', 2, 'LineStyle', styles{idri}); 
    plot(drange, mean(pck(:,:),2), 'LineWidth', 2, 'LineStyle', styles{idri});
    axis([0 0.5 0 1]); grid on; hold all;
end
legend(texts, 'Location', 'SouthEast');
axis equal;
axis square;
set(gca, 'FontSize', 16);
marks = 0:0.1:1;
set(gca, 'XTick', marks);
tlabels = {};
for il=1:length(marks)
    tlabels{il} = sprintf('%0.1f', marks(il));
%     if mod(il,2)==0
%         tlabels{il} = '';
%     end
end
% axis([0 0.5 0 1]);
% set(gca, 'XTickLabel', tlabels);
axis square;
% set(gca, 'YTick', 0:0.1:1);
% export_fig -a1 -m1 ./pck_iters.pdf
% legend off;
% export_fig -a1 -m1 ./pck_iters_noleg.pdf
% axis([0 0.5 0.5 1]);
% export_fig -a1 -m1 ./pck_iters_zoom1.pdf
