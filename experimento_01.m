% =========================================================================
% ETAPA 1: LEITURA E PREPARAÇÃO DO DATASET GIGANTE (9.987 CLASSES)
% =========================================================================
clear; clc; close all;

nomeArquivo = 'banco_completo_treinamento_rna.csv'; 
disp('Carregando a base de dados GIGANTE... Isso pode levar alguns minutos.');

tabelaDados = readtable(nomeArquivo);

% Extraindo as entradas (Lat_P, Long_P, Lat_C, Long_C)
input = tabelaDados{:, 1:4};

% Extraindo a Saída para Categorical (Blindagem contra estouro de RAM)
classes = tabelaDados{:, 5};
output_cat = categorical(classes);

numAmostras = length(classes);
numClasses = numel(categories(output_cat));

disp(['Total de amostras carregadas: ', num2str(numAmostras)]);
disp(['Total de Classes (Rotas): ', num2str(numClasses)]);
disp('----------------------------------------------------');

% Divisão 70% Treino, 15% Validação, 15% Teste
cv = cvpartition(numAmostras, 'HoldOut', 0.30);
idxTreino = training(cv);
idxTemp = test(cv);

cvTest = cvpartition(sum(idxTemp), 'HoldOut', 0.50);
idxTestLogico = false(sum(idxTemp), 1);
idxTestLogico(test(cvTest)) = true;

idxVal = false(numAmostras, 1);
idxTest = false(numAmostras, 1);
tempIndices = find(idxTemp);
idxVal(tempIndices(~idxTestLogico)) = true;
idxTest(tempIndices(idxTestLogico)) = true;

XTreino = input(idxTreino, :); YTreino = output_cat(idxTreino);
XVal    = input(idxVal, :);    YVal    = output_cat(idxVal);
XTeste  = input(idxTest, :);   YTeste  = output_cat(idxTest);

% =========================================================================
% ETAPA 2: ARQUITETURA DA REDE NEURAL (TESTE DE CONTROLE - 130 NEURÔNIOS)
% =========================================================================
layers = [
    featureInputLayer(4, 'Normalization', 'zscore', 'Name', 'Entrada_Normalizada')
    
    % MANTIDO EM 130 NEURÔNIOS para o teste comparativo de gargalo espacial
    fullyConnectedLayer(130, 'Name', 'Camada_Escondida_130_Neuronios')
    tanhLayer('Name', 'Ativacao_Tansig_Original') 
    
    fullyConnectedLayer(numClasses, 'Name', 'Mapeamento_Para_Classes')
    softmaxLayer('Name', 'Probabilidades')
    classificationLayer('Name', 'Classificador_CrossEntropy')
];

% Lote de 4096: Isola o ruído estocástico, mantendo a descida estável
tamanhoLote = 4096; 
freqValidacao = max(1, floor(length(YTreino) / tamanhoLote));

options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'auto', ... 
    'MaxEpochs', 30000, ...               % Teto máximo para garantir a convergência total
    'InitialLearnRate', 0.001, ...        % Taxa padrão e segura do Adam, sem ruídos
    'MiniBatchSize', tamanhoLote, ...             
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', freqValidacao, ... % Valida exatamente 1x por época
    'ValidationPatience', 30, ...         % AUMENTADO: Super paciência para suportar as 9.987 classes
    'Plots', 'training-progress', ...
    'Verbose', false);

disp(['Treinando a RNA de 130 neurônios na base gigante com Lote de 4096... Acompanhe o gráfico!']);
[netRNA, info] = trainNetwork(XTreino, YTreino, layers, options);

% =========================================================================
% ETAPA 3: CÁLCULO DE MÉTRICAS AVANÇADAS E RELATÓRIO
% =========================================================================
disp('Treinamento finalizado. Classificando dados para gerar o relatório...');

YPred_Treino = classify(netRNA, XTreino, 'MiniBatchSize', tamanhoLote);
YPred_Val    = classify(netRNA, XVal, 'MiniBatchSize', tamanhoLote);
YPred_Teste  = classify(netRNA, XTeste, 'MiniBatchSize', tamanhoLote);
YPred_All    = classify(netRNA, input, 'MiniBatchSize', tamanhoLote);

Acc_Train = (sum(YPred_Treino == YTreino) / numel(YTreino)) * 100;
Acc_Val   = (sum(YPred_Val == YVal) / numel(YVal)) * 100;
Acc_Test  = (sum(YPred_Teste == YTeste) / numel(YTeste)) * 100;
P_general = (sum(YPred_All == output_cat) / numel(output_cat)) * 100;

% Cálculo de Macro-Precision e Macro-Recall 
C = confusionmat(YTeste, YPred_Teste);
diagC = diag(C);
sumLinhas = sum(C, 2);   
sumColunas = sum(C, 1)'; 

precisions = diagC ./ sumColunas;
recalls = diagC ./ sumLinhas;
precisions(isnan(precisions)) = 0;
recalls(isnan(recalls)) = 0;

Macro_Precision = mean(precisions) * 100;
Macro_Recall = mean(recalls) * 100;

% Cálculo matemático robusto da Melhor Época
valIdx = find(~isnan(info.ValidationLoss));
[minValLoss, minIdx] = min(info.ValidationLoss(valIdx));
bestIter = valIdx(minIdx);
bestEpoch = ceil(bestIter / freqValidacao); 

fprintf('\n====================================================\n');
fprintf('RELATÓRIO FINAL - 130 NEURÔNIOS (9.987 CLASSES)\n');
fprintf('====================================================\n');
fprintf('Fase\t\tAmostras\tAcurácia (%%)\n');
fprintf('Treino\t\t%d\t\t%.2f\n', numel(YTreino), Acc_Train);
fprintf('Validação\t%d\t\t%.2f\n', numel(YVal), Acc_Val);
fprintf('Teste\t\t%d\t\t%.2f\n', numel(YTeste), Acc_Test);
fprintf('----------------------------------------------------\n');
fprintf('ACURÁCIA GERAL:    %.2f%%\n', P_general);
fprintf('MACRO-PRECISION:   %.2f%%\n', Macro_Precision);
fprintf('MACRO-RECALL:      %.2f%%\n', Macro_Recall);
fprintf('----------------------------------------------------\n');
fprintf('MELHOR ÉPOCA:      %d\n', bestEpoch);
fprintf('====================================================\n');

% =========================================================================
% ETAPA 4: GERAÇÃO DOS GRÁFICOS
% =========================================================================
disp('Gerando gráficos otimizados para a tese...');

figure('Name', 'Best Validation Performance', 'NumberTitle', 'off');
plot(info.TrainingLoss, 'b', 'LineWidth', 1.5); hold on;
plot(valIdx, info.ValidationLoss(valIdx), 'k--', 'LineWidth', 1.5);
plot(bestIter, minValLoss, 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', ['Melhor Desempenho (Época ', num2str(bestEpoch), ')']);
title(['Melhor Desempenho de Validação: ', num2str(minValLoss, '%.4f'), ' na Época ', num2str(bestEpoch)]);
xlabel('Iterações'); ylabel('Cross-Entropy Loss');
legend('Treino', 'Validação', 'Melhor Validação'); grid on;

tind_teste = double(YTeste)';
yind_teste = double(YPred_Teste)';
figure('Name', 'Alvos vs Predições (Teste)', 'NumberTitle', 'off');
plot(tind_teste, 'b.', 'MarkerSize', 8, 'DisplayName', 'Alvo Real'); 
hold on;
plot(yind_teste, 'r.', 'MarkerSize', 4, 'DisplayName', 'Predição RNA');
title('Dispersão: Alvo vs Predição (Conjunto de Teste)');
xlabel('Índice da Amostra'); ylabel('Classe Predita');
legend; grid on;

probs = predict(netRNA, XTeste, 'MiniBatchSize', tamanhoLote);
confianca_max = max(probs, [], 2); 
acertos = (YPred_Teste == YTeste);
figure('Name', 'Histograma de Confiança de Predição', 'NumberTitle', 'off');
histogram(confianca_max(acertos), 'FaceColor', 'g', 'BinWidth', 0.05, 'DisplayName', 'Acertos'); hold on;
if sum(~acertos) > 0
    histogram(confianca_max(~acertos), 'FaceColor', 'r', 'BinWidth', 0.05, 'DisplayName', 'Erros');
end
title('Distribuição de Confiança (Erros e Acertos)');
xlabel('Grau de Confiança (0.0 a 1.0)'); ylabel('Frequência (Amostras)');
legend; grid on;

figure('Name', 'Resumo de Acurácias', 'NumberTitle', 'off');
bar([Acc_Train, Acc_Val, Acc_Test, P_general], 'FaceColor', [0.1 0.5 0.3]);
set(gca, 'XTickLabel', {'Treino', 'Validação', 'Teste', 'Geral'});
ylabel('Acurácia (%)'); ylim([0 100]);
title('Visão Geral de Acurácia da Rede');
grid on;

% =========================================================================
% ETAPA 5: SALVAMENTO SEGURO (TUDO DO WORKSPACE)
% =========================================================================
% Omitindo os nomes das variáveis, o comando salva o Workspace inteiro
save('Workspace_RNA_9987_Classes_130_Neuronios_Batch4096.mat', '-v7.3');
disp('Treinamento finalizado, métricas calculadas e todo o Workspace salvo com sucesso!');