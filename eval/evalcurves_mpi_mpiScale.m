% Testing result directory
seq_name = 'hand_mpi_nzl_v20_orig_ts';

% Data is kept in json format
addpath(genpath('/media/posefs1b/Users/tsimon/loop/devel/panoptic-toolbox/matlab'));
% addpath(genpath('./util/ojwoodford-export_fig-5735e6d'));
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

% Models to evaluate
models = {
    'hands_v143_noft_102k',
    'my_measure',
    'CPM',
    'snapshots_cpm_rotate_s10_wrist_dome_simon'};

close all;
clear results;
do_plot = 0;
lentest = nan;
for idm = 1:length(models)
    idm
%     if idm > 6
%         do_plot=1;
%     end
    model_name = models{idm};
    
    test_files = {};
    warning off;
    % in_path = sprintf('/media/posefs3b/Users/tsimon/outputs/hands/%s/%s/mat/', seq_name, model_name);
    if idm <= 1
        in_path = sprintf('/media/posefs3b/Users/tsimon/outputs/hands/%s/%s/mat/', seq_name, model_name);
    else
        in_path = sprintf('./%s/%s/mat/', seq_name, model_name);
    end
    % for iddirs=1:length(test_dirs)
%     Dfiles = dir( fullfile(in_path, '*.mat_l.json') );
%     if ~isempty(Dfiles)
%         for idf=1:length(Dfiles)
%             fname = Dfiles(idf).name;
%             fname_out = fname(1:end-7);
%             movefile(fullfile( in_path, fname), fullfile(in_path,fname_out));
%         end
%     end
    Dfiles = dir( fullfile(in_path, '*.mat') );
    for idf=1:length(Dfiles)
        test_files = {test_files{:}, fullfile(in_path, Dfiles(idf).name) };
    end
    
    %
    if idm==1
        lentest=length(test_files);
    else
        assert(lentest==length(test_files));
    end

    all_dist = zeros(21, length(test_files));
    all_types = zeros(length(test_files),1);
    for idf=1:length(test_files)
%         idf
        hentry = load(test_files{idf});
        test_imagen = sprintf('%s', hentry.originim);
        test_image = test_imagen;
        
        if ~strcmp(hentry.ver, 'mpi')
%             continue;
%             error('This doesnt work on non-mpi');
            is_mpi = 0;
            [~,nm,~] = fileparts(test_image);
            posefile = sprintf('%s/%s.json', '/media/posefs0c/Users/donglaix/nzl_images2-pose/', ...
                nm);
            pose = loadjson(posefile);
            joints = reshape(pose.bodies{1}.joints, 3, []);
            if any(any(joints(1:2,1:2)==0))
                error('joints');
            end
            if any(joints(3,1:2)<0.1)
                error('confidence');
            end
            hs = abs(joints(2,1)-joints(2,2));
            lwt_annot.x1 = joints(1,1)-hs/3;
            lwt_annot.x2 = joints(1,1)+hs/3;
            lwt_annot.y1 = joints(2,1);
            lwt_annot.y2 = joints(2,2);
        else
%             continue;
            is_mpi = 1;
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
            if is_mpi
                plot([hentry.annot.annorect(id_person).annopoints.point.x], [hentry.annot.annorect(id_person).annopoints.point.y], 'c*');
            else 
                plot(joints(1,:), joints(2,:), 'c*');
            end
            drawnow;
%             pause;
        end
        
        if is_mpi
            mpi_annot = hentry.annot.annorect;
            if length(hentry.annot.annorect)>1
                %Check the right one
                distFromWrist=[];
                for pIdx = 1:length(hentry.annot.annorect)
                    if(isempty(hentry.annot.annorect(pIdx).annopoints) || length(hentry.annot.annorect(pIdx).annopoints.point)<16)
                        distFromWrist(pIdx) = 1e5;
                        continue;
                    end
                    
                    lwrist = [hentry.annot.annorect(pIdx).annopoints.point(16).x hentry.annot.annorect(pIdx).annopoints.point(16).y];
                    rwrist = [hentry.annot.annorect(pIdx).annopoints.point(11).x hentry.annot.annorect(pIdx).annopoints.point(11).y];
                    distFromWrist(pIdx) = min(norm(hentry.bbcenter(1:2) - lwrist),norm(hentry.bbcenter(1:2) - rwrist));
                end
                [bestVal bestIdx] = min(distFromWrist);
%                 bestVal
                if bestVal>100
                    bestVal
                    pause;
                end
                mpi_annot = hentry.annot.annorect(bestIdx);
                
            end
            headSize = abs(mpi_annot.y1-mpi_annot.y2);  %MPI dist
            if do_plot
                rectangle('Position',[mpi_annot.x1 mpi_annot.y1 abs(mpi_annot.x1-mpi_annot.x2) abs(mpi_annot.y1-mpi_annot.y2)],'edgecolor','r');
%                 pause;
            end
        else
            headSize = abs(lwt_annot.y1-lwt_annot.y2);
            if do_plot
                rectangle('Position',[lwt_annot.x1 lwt_annot.y1 abs(lwt_annot.x1-lwt_annot.x2) abs(lwt_annot.y1-lwt_annot.y2)],'edgecolor','r');
%                 pause;
            end
        end
        drawnow;
        % Scale normalization: this needs to change
        dist = sqrt(sum((hentry.hand_pts(:,1:2)-prediction(:,1:2)).^2,2));
        dist_scaleByBBox = dist/(hentry.bbsqsize);   %Original version
        
        headSize = 0.7*headSize;
        dist = dist/headSize;
        %dist =dist_scaleByBBox;
        
        all_dist(:, idf) = dist;
        all_types(idf) = 1;%s
        
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
            %            sel_files = result.all_types==1;
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
sel_results = 1:4;
% texts = {'Render (initial)', 'Render (iteration 2)', 'Manual (initial)', 'Manual (iteration 2)', 'Mix (initial)', 'Mix (iteration 3)', 'CPM'};
texts = {'Tomas reported', 'My measured', 'My trained', 'My trained (with dome data)'};
% styles = {':', '-', ':', '-', ':', '-', ':','-',':'}; 
styles = {':', '-', ':', '-'};
% cols =[    0.8941    0.1020    0.1098
%     0.3020    0.6863    0.2902
%     0.2157    0.4941    0.7216
%     ];
% cols = kron(cols, [0.85; 1]);
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
%             assert(sum(sel_files)==size(result.all_dist,2));
            d = squeeze(result.all_dist(jidx,sel_files));

            % dist is NaN if gt is missing; ignore dist in this case
            pck(k,jidx) = pck(k,jidx) + mean(d(~isnan(d))<=drange(k));
        end
    end
    disp(mean(mean(pck)));
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
grid on;
%%% mkdir /media/posefs3b/Users/tsimon/mvbs/scaleCR_nzsl
%axis([0 0.5 0.5 1]);
% set(gca, 'XTickLabel', tlabels);
axis square;
set(gca, 'YTick', 0:0.1:1);
% export_fig -a1 -m1 /media/posefs3b/Users/tsimon/mvbs/scaleCR_nzsl/pck_iters.pdf
% legend off;
% export_fig -a1 -m1 /media/posefs3b/Users/tsimon/mvbs/scaleCR_nzsl/pck_iters_noleg.pdf
% axis([0 0.5 0.5 1]);
% export_fig -a1 -m1 /media/posefs3b/Users/tsimon/mvbs/scaleCR_nzsl/pck_iters_zoom1.pdf
