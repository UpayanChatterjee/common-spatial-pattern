function [result] = f_classifier(feature, label)

    coeff = lda(feature, label);
    X_lda = feature * coeff;

    %plot the projected data
    gscatter(X_lda(:,1),X_lda(:,2),label,'rb','.');
    xlabel('LD1');
    ylabel('LD2');
    legend('Left','Right');
    title('LDA on EEG Data');
end