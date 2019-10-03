function[] = nCorrToDavis( dicdata )
%Save an nCorr DIC array into a DaVis-type txt file for DefDAP

outputFileName = 'ncorrexport.txt';

%Check size of DIC array and subset size
sz = size(dicdata.data_dic.displacements.plot_u_dic);
x_len = sz(2);
y_len = sz(1);
subset_size = dicdata.data_dic.dispinfo.radius;

%Make header string
header = ['#nCorr 1.2 2D-vector ', int2str(subset_size), ' ', int2str(y_len), ' ', int2str(x_len), ' "" "pixel" "" "pixel" "displacement" "pixel"\n'];

%Print header to file
fileID = fopen(outputFileName,'w');
fprintf(fileID,header);

u=dicdata.data_dic.displacements.plot_u_dic;
v=dicdata.data_dic.displacements.plot_v_dic;

%Print data to file
for y=1:y_len
    for x=1:x_len
        fprintf(fileID,'%d\t%d\t%6.6f\t%6.6f\t\n',[((x-1)*subset_size)+round(subset_size/2),((y-1)*subset_size)+round(subset_size/2),u(y,x),v(y,x)]);
    end
end

%Close file
fclose(fileID);

end

