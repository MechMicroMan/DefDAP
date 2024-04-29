function[] = nCorrToDavis(dicdata, outputFileName)
% Saves an nCorr DIC array into a DaVis-type txt file for DefDAP
 
% For example, if your nCorr data is in a variable called 'ans'
% and you want to export to a file called 'export.txt'
% run:
% nCorrToDavis(ans, 'export.txt')
 
%Check size of DIC array and subset size
sz = size(dicdata.data_dic.displacements.plot_u_dic);
x_len = sz(2);
y_len = sz(1);

% subset size
subset_size = dicdata.data_dic.dispinfo.radius * 2;
 
% subset spacing
subset_spacing = dicdata.data_dic.dispinfo.spacing;
 
%Make header string
header = ['#nCorr 1.2 2D-vector ', ...
    int2str(subset_spacing), ...
    ' ', ...
    int2str(y_len), ...
    ' ', ...
    int2str(x_len), ...
    ' "" "pixel" "" "pixel" "displacement" "pixel"\n'];
 
%Print header to file
fileID = fopen(outputFileName,'w');
fprintf(fileID,header);
 
u=dicdata.data_dic.displacements.plot_u_dic;
v=dicdata.data_dic.displacements.plot_v_dic;
 
f = waitbar(0, 'Please wait...');
 
%Print data to file
for y=1:y_len
    waitbar(y/y_len, f, 'Exporting data to txt file...')
    for x=1:x_len
        fprintf(fileID,'%1.1f\t%1.1f\t%6.6f\t%6.6f\n', ...
        [((x-1)*subset_spacing)+subset_spacing./2, ...
            ((y-1)*subset_spacing)+subset_spacing./2, ...
            u(y,x), ...
            v(y,x)]);
    end
end
 
close(f)
 
%Close file
fclose(fileID);
 
disp(['Exported to: ', outputFileName])
 
end
