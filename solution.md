# **2025数据科学导论 Project 2：利用预训练模型和无标注数据的预测驱动统计推断**

## **摘要**

在当前数据驱动的科研与工业应用中，大规模未标注数据和高性能预训练机器学习模型日益普及，然而获取充足的高质量标注数据以进行可靠的统计推断仍然面临成本高昂和耗时费力的挑战。本项目旨在应对这一挑战，提出并评估一种结合少量有价值的标注数据、海量低成本的未标注数据以及预训练机器学习模型预测结果的统计推断框架。核心方法借鉴并扩展了预测驱动推断（Prediction-Powered Inference, PPI）的思想，特别是其增强版PPI++，该方法通过精心设计的校正机制，利用标注数据量化并修正机器学习模型的预测偏差，从而在不依赖于机器学习模型具体结构的前提下，实现对目标参数（如均值、分位数、回归系数）的有效推断 1。本研究将通过Python 3.12环境，借助Cursor AI编辑器进行代码实现与实验分析，系统评估所选PPI++方法在提升统计推断效率（例如，获得更窄且有效的置信区间）方面的表现。评估将与两种基准方法进行对比：一种是直接使用机器学习模型的预测值进行推断（朴素推断），另一种是仅依赖少量标注数据进行传统统计推断。研究结果预期将表明，预测驱动的统计推断方法能够在保证统计有效性的同时，显著提升数据利用效率和推断精度，为在标注数据稀缺情景下进行科学发现和决策提供一种更为强大和可靠的途径。

## **1\. 引言**

### **1.1. 背景与动机**

随着信息技术的飞速发展，各行各业积累了海量的未标注数据，同时，功能强大的预训练机器学习（ML）模型，如各类深度学习网络，也变得唾手可得 1。这些模型能够从未标注数据中学习复杂的模式，并对新数据进行预测。然而，在许多科学研究和商业决策场景中，我们不仅需要预测，更需要对总体参数进行可靠的统计推断，例如估计某个效应的大小、检验某个假设的真伪等。传统的统计推断方法通常依赖于充足的高质量标注数据，但标注过程往往成本高昂、耗时费力，甚至在某些领域（如医学影像、遥感分析）难以大规模实施 4。

半监督学习（Semi-supervised learning, SSL）作为一种机器学习范式，试图利用少量标注数据和大量未标注数据进行学习 5，这为解决上述数据稀缺问题提供了一个总体思路。然而，如何将SSL的思想与严格的统计推断相结合，确保推断结果的有效性（如置信区间的覆盖率、假设检验的I类错误率），仍然是一个活跃的研究领域。

### **1.2. 问题陈述**

当仅有少量标注数据可用时，经典的统计推断方法（例如t检验、线性回归等）往往面临诸多局限。样本量不足会导致统计功效低下，难以检测到真实的效应；参数估计的方差较大，导致置信区间过宽，缺乏实用价值；同时，基于小样本的p值也可能不够可靠，容易产生错误的结论 7。

另一方面，虽然预训练ML模型可以为大量未标注数据提供预测值，但若将这些预测值（通常是“黑箱”模型的输出）不加甄别地直接视为真实标签或结果，并用于后续的统计分析，则会引入严重的统计问题。研究表明，这种朴素的“即插即用”方法（plug-in estimation）会导致参数估计产生偏差、方差估计不准确，进而使得构建的置信区间覆盖率失准，p值失效，最终可能得出完全错误的统计结论 13。正如Angelopoulos等人在其开创性工作中指出的，即使ML预测的准确率高达80-90%，直接使用这些预测进行下游回归分析也可能导致严重的偏差和无效的置信区间 1。其根本原因在于，ML模型的预测本身存在误差和潜在偏倚，这些不确定性若未在统计推断中得到妥善处理，将严重损害推断的有效性。

### **1.3. 项目目标**

本项目的主要目标是实现并严格评估一种先进的统计建模方法，该方法能够有效地结合以下三方面信息进行有效的统计推断：

1. 少量高质量的标注数据集。  
2. 大量易于获取的未标注数据集。  
3. 由预训练机器学习模型提供的关于未标注数据的预测结果。

具体的次级目标包括：

* 将所选的预测驱动推断方法与以下两种基准方法进行比较：  
  1. 直接使用机器学习模型的预测结果作为真实值进行统计推断（朴素推断）。  
  2. 仅使用少量标注数据进行经典的统计推断。  
* 使用Python 3.12编程语言和Cursor AI代码编辑器完成所有方法的实现和实验评估。  
* 撰写一份可复现的研究报告，包含Markdown（MD）和LaTeX两种格式，详细记录项目的方法、过程、结果与讨论。

### **1.4. 范围与贡献**

本项目将聚焦于特定类型的统计推断任务，例如总体均值的估计、总体分位数的估计，或线性回归模型中特定系数的估计，并为这些估计量构建置信区间。所选用的核心方法将是预测驱动推断（Prediction-Powered Inference, PPI）框架的一个变体，很可能是PPI++，因为它在效率和自适应性方面具有优势。

本项目的贡献在于提供一个在常见数据科学场景下（即标注数据稀缺但存在未标注数据和预训练模型）应用和评估预测驱动推断方法的实用指南和实证研究。通过清晰展示其相对于传统方法和朴素ML预测应用的优势，本项目旨在为数据科学从业者和研究者提供一种更可靠、更高效地利用现有数据资源进行统计推断的思路和工具。

### **1.5. 报告结构**

本报告的后续章节安排如下：第二部分将详细介绍预测驱动统计推断的理论框架，包括其基本原理、核心组件（如校正项）以及所选PPI++方法的特点。第三部分将阐述实验设计和具体实现细节，包括数据集的选择与预处理、预训练模型的规格、三种对比推断方法的实现方案，以及开发环境和所用工具。第四部分将呈现实验结果和评估，重点比较不同方法在点估计偏差、均方误差、置信区间覆盖率和长度等方面的表现。第五部分将对实验结果进行深入讨论，解读其含义，分析研究的局限性，并展望未来工作。第六部分为结论，总结本研究的主要发现和意义。第七部分将提供关于项目可复现性的说明。最后是参考文献和可能的附录。

在现代数据科学实践中，一个核心的挑战在于如何弥合机器学习强大的预测能力与经典统计学对推断结论严格有效性要求之间的鸿沟。机器学习模型，特别是大规模预训练模型，能够从未标注数据中学习并生成海量的预测信息，但这些预测本身并不直接提供统计学意义上的不确定性量化。另一方面，传统的统计推断方法虽然能提供严格的有效性保证（如置信区间的正确覆盖率），但在标注数据稀缺时其效能大打折扣。预测驱动推断（PPI）及其变体，正是为了解决这一核心矛盾而提出的。它们不试图替代统计推断，而是旨在将机器学习的预测信息以一种统计学上稳健的方式整合到推断过程中，从而在保证有效性的前提下，提升推断的效率和精度。这一框架的一个显著优点在于其对机器学习模型的“不可知论（agnosticism）” 1：它不要求深入了解预训练模型的内部结构或训练细节，仅利用其输出的预测值，并通过少量标注数据来校正这些预测中可能存在的系统性偏差。这种特性使得PPI方法具有广泛的适用性，尤其是在用户可能无法控制或完全理解所用预训练模型的场景中。

## **2\. 预测驱动的统计推断框架**

### **2.1. 预测驱动推断（PPI）基础**

预测驱动推断（Prediction-Powered Inference, PPI）是一种新兴的统计框架，其核心思想在于巧妙地结合两类数据源的优势：一类是数量庞大但可能存在未知偏差的机器学习（ML）模型预测结果，另一类是数量稀少但质量可靠的“金标准”标注数据 1。PPI的目标是利用这两类信息，对感兴趣的总体参数（如均值、分位数、回归系数等）进行有效的统计推断，即构建具有预设覆盖率的置信区间或进行假设检验。

该框架的一般流程是：首先，利用预训练的ML模型 f(X) 对大量的未标注数据 X1′​,...,XN′​ 产生预测值 Y^1′​,...,Y^N′​。这些预测值被用来形成对目标参数的一个初步的、可能是有偏的估计。然后，利用小规模的标注数据集 (X1​,Y1​),...,(Xn​,Yn​)，其中 Yi​ 是真实的标签，而 Y^i​=f(Xi​) 是ML模型对这些已标注样本的预测。通过比较 Yi​ 和 Y^i​，PPI能够量化ML模型的预测误差，并构建一个所谓的“校正项”（rectifier）。这个校正项随后被用来调整基于大量预测值得出的初步估计，从而得到一个经过偏差校正的、更可靠的最终估计量 1。

PPI框架的普适性很强，它可以应用于任何能够表示为凸目标函数最小化器的参数估计问题 1。这意味着除了常见的均值、分位数和回归系数外，许多其他复杂的统计量也可以通过PPI进行推断。一个关键特性是，PPI对提供预测的ML模型 f(X) 本身不作任何假设，无论是模型的结构、训练方法还是其预测的准确度如何，PPI都能保证最终推断的统计有效性 1。

### **2.2. 校正项（Rectifier）在偏差修正中的作用**

校正项是PPI框架中实现偏差修正的核心机制。其基本思想是利用已有的少量标注数据来估计和补偿由ML模型预测引入的系统性偏差。校正项的具体数学形式取决于所要估计的目标参数 θ 以及相应的“拟合度量”（measure of fit） mθ​。

在Angelopoulos等人 1 的原始工作中，对于一个候选的参数值 θ，拟合度量 mθ​ 在基于预测值 Y^′ 的未标注数据集上计算，用于量化 θ 基于这些（可能不完美的）预测数据的合理性。校正项 Dθ​ (或 Δθ​) 则被定义为在标注数据集上，真实标签 Y 计算的拟合度量与预测标签 Y^ 计算的拟合度量之差。例如，在估计总体均值 μ=E\[Y\] 的简单情况下，一个朴素的基于预测的估计是 N1​∑i=1N​f(Xi′​)。校正项可以理解为对 f(X) 预测偏差的经验估计，通常形式为 n1​∑i=1n​(f(Xi​)−Yi​)，即模型预测值与真实值在标注样本上的平均差异 18。

最终的PPI估计量 θ^PPI​ 通常是将基于大量预测的初步估计量（例如 N1​∑f(Xi′​)）减去这个校正项（或其某种形式的推广）得到。例如，对于均值估计，PPI估计量可以表示为 18：  
μ^​PPI​=N1​i=1∑N​f(Xi′​)−n1​j=1∑n​(f(Xj​)−Yj​)  
这个校正步骤确保了 θ^PPI​ 是目标参数 θ 的一个（渐进）无偏估计量。通过这种方式，即使ML模型的预测 f(X) 存在系统性偏差，只要这个偏差能够在标注数据上被稳定地估计出来，PPI就能够有效地消除它对最终统计推断的影响，从而保证置信区间的有效覆盖率和假设检验的I类错误控制。

### **2.3. 所选PPI变体：PPI++ 以增强效率和自适应性**

尽管原始PPI框架具有开创性和普适性，但在某些情况下，其计算效率和统计效率可能并非最优。为了解决这些问题，Angelopoulos等人后续提出了PPI++ 19。PPI++ 的核心改进在于引入了一个称为“功效调整参数”（power-tuning parameter）的 λ∈R。这个参数用于自适应地权衡经典统计估计量（仅基于标注数据）和利用ML预测进行校正的部分。

PPI++的估计量通常基于一个调整后的损失函数 LPPλ​(θ)，其形式为 20：  
LPPλ​(θ):=Ln​(θ)+λ⋅(LfN​(θ)−Lfn​(θ))  
其中，Ln​(θ) 是基于标注数据的经典损失函数，LfN​(θ) 是基于大量未标注数据（使用 f(X) 的预测值）的损失函数，Lfn​(θ) 是基于标注数据（但使用 f(X) 的预测值而非真实 Y）的损失函数。参数 λ 控制了预测信息（即 LfN​(θ)−Lfn​(θ) 这一项，它反映了预测在未标注数据上带来的“增益”相对于其在标注数据上的“表现”）在多大程度上被纳入最终估计。  
λ 的选择至关重要。PPI++的一个关键特性是，λ 可以从数据中估计得到，目标通常是最小化最终PPI++估计量的渐进方差或最大化相关假设检验的功效 19。Python库 ppi\_py 在其实现中，允许用户不指定 lam 参数，此时库会自动从数据中估计最优的 λ 值 22。当 λ=0 时，PPI++估计量退化为仅使用标注数据的经典估计量；当 λ=1 时，在某些特定情况下（例如，损失函数具有特定形式），PPI++可能恢复到原始PPI估计量的形式 22。

这种自适应性是PPI++相较于原始PPI的一个重要优势。它使得方法能够根据预训练模型 f(X) 的实际预测质量自动调整。如果 f(X) 的预测非常准确，估计出的 λ 可能会接近1，从而充分利用大量的预测信息以减小估计方差、缩短置信区间。反之，如果 f(X) 的预测质量较差或与真实值关联不大，估计出的 λ 可能会趋向于0，此时PPI++将更多地依赖于可靠的标注数据，表现接近于经典推断方法，从而避免了被劣质预测误导的风险。这种稳健性使得PPI++在实际应用中更具吸引力，因为它确保了（在渐进意义下）其表现至少不劣于仅使用标注数据的经典方法，并且在预测信息有用时能显著提升推断效率 20。

### **2.4. 与预训练模型 f(X) 的集成**

PPI/PPI++框架与预训练模型 f(X) 的集成方式非常直接且灵活。预训练模型 f(X) 的角色是为特征向量 X 提供预测值 Y^。这些预测值将在两个关键环节被使用：

1. **在标注数据集上**：对于每一个标注样本 (Xj​,Yj​)，模型会产生预测 Y^j​=f(Xj​)。这些预测值 Y^j​ 与真实标签 Yj​ 一同用于计算校正项，如前文所述。  
2. **在大量未标注数据集上**：对于每一个未标注样本 Xi′​，模型会产生预测 Y^i′​=f(Xi′​)。这些预测值 Y^i′​ 被用来构建对目标参数的初步估计，该估计随后会被校正项调整。

PPI/PPI++框架的一个核心优势在于其对预训练模型 f(X) 的“黑箱”处理方式 1。这意味着研究者无需关心 f(X) 的具体架构（例如，是线性模型、决策树、神经网络还是更复杂的集成模型）、训练细节（例如，训练数据、优化算法、超参数设置）或其内部工作原理。只要模型能够输入特征 X 并输出对目标变量 Y 的预测 Y^，它就可以被整合到PPI框架中。这种模型无关性极大地增强了PPI方法的实用性和普适性，使得研究者可以方便地利用已有的、可能是由第三方提供的、或者结构非常复杂的预训练模型，而无需对其进行修改或深入分析。

### **2.5. 处理潜在的分布偏移（简述）**

在实际应用中，标注数据和未标注数据的来源或收集条件可能存在差异，导致它们的概率分布不完全一致，即存在分布偏移（distribution shift）。PPI框架及其变体也考虑了这类更具挑战性的情况，并提供相应的解决方案，以保持推断的有效性 1。

主要考虑的分布偏移类型包括：

* **协变量偏移（Covariate Shift）**：指的是特征 X 的边际分布在标注数据集和未标注数据集之间发生变化（即 Plab​(X)=Punlab​(X)），但条件分布 P(Y∣X) 保持不变。在这种情况下，PPI可以通过对损失函数或拟合度量进行适当的重加权（reweighting）来处理。如果能够估计或已知两个分布之间的密度比 w(x)=Punlab​(X=x)/Plab​(X=x)，则可以将对未标注数据分布上参数的估计问题，转化为在标注数据分布上对一个加权损失函数的优化问题 1。例如，损失函数 Lθ​(x,y) 变为 w(x)Lθ​(x,y)。  
* **标签偏移（Label Shift）**：指的是目标变量 Y 的边际分布在标注数据集和未标注数据集之间发生变化（即 Plab​(Y)=Punlab​(Y)），但条件分布 P(X∣Y) 保持不变。这在分类问题中较为常见。PPI可以通过估计混淆矩阵（confusion matrix）并在标注数据和未标注数据的预测分布之间建立联系来处理标签偏移，从而调整对目标参数（如类别比例）的估计 1。

尽管对分布偏移的详细处理可能超出了本入门级数据科学项目的核心范围，除非所选数据集明确表现出此类特性，但了解PPI框架具备处理这些复杂情况的能力，进一步证明了其理论的完备性和实践的广泛适用性。在存在分布偏移的情况下，PPI协议依然力求保持其统计有效性，并利用机器学习预测来提升统计功效 1。

## **3\. 实验设置与实现细节**

### **3.1. 数据集选择与预处理**

#### **3.1.1. 数据集选择**

为了有效地评估和比较不同的统计推断方法，选择一个合适的数据集至关重要。本研究将采用公开可用的**阿尔茨海默病神经影像学倡议（Alzheimer's Disease Neuroimaging Initiative, ADNI）数据库** 24。ADNI数据库是一个大规模、多中心的纵向研究项目，旨在收集和共享用于研究阿尔茨海默病（AD）早期检测和进展的临床、影像、遗传和生物标记物数据。该数据集因其数据的丰富性、多模态性以及在众多医学研究中的广泛应用而著称，并且已被用于与预测驱动推断相关的研究中 24。

选择ADNI数据集的理由如下：

1. **相关性**：ADNI数据涉及复杂的生物医学问题，其中参数估计（例如，认知评分的平均变化、某种生物标记物与疾病状态的关联强度）具有实际临床意义。  
2. **数据可用性**：ADNI数据对研究人员开放，便于获取和复现。  
3. **适用性**：ADNI数据集中包含多种类型的变量（如人口学信息、临床评分、基因数据、影像特征等），可以灵活选择特征 X 和目标结果 Y。例如，我们可以选择一组基线协变量 X（如年龄、APOE4基因型、教育程度、基线MMSE评分）来预测某个时间点后的认知变化（如ADAS-Cog评分的变化量）或疾病进展状态 Y。  
4. **模拟“小标注、大未标注”场景**：ADNI数据集规模较大，可以从中划分出一个小规模的“标注”子集（包含完整的 X 和 Y）和一个大规模的“未标注”子集（仅使用 X，其对应的 Y 在PPI方法中被视为未知，但可用于评估）。

#### **3.1.2. 数据划分策略**

从ADNI数据集中，将选取符合特定入组标准的受试者数据。假设原始可用样本总数为 Ntotal​。数据将按以下方式划分：

1. **预训练模型训练/验证集（可选）**：如果需要自行“预训练”一个机器学习模型 f(X)（而非直接使用外部预训练模型），将从 Ntotal​ 中划分一部分数据，例如 Npretrain​ 个样本，专门用于训练和验证该模型。这部分数据将与后续用于PPI校正和推断的数据完全分离，以避免信息泄露。  
2. **小规模标注集 (Xlab​,Ylab​)**：从剩余数据中随机抽取 nlab​ 个样本作为标注集。这个集合将包含完整的特征 Xlab​ 和真实结果 Ylab​。nlab​ 的选择将反映“少量标注”的现实情况，例如，根据研究目标和数据特性，可设为100到500之间。  
3. **大规模未标注特征集 Xunlab​**：从剩余数据中再随机抽取 Nunlab​ 个样本的特征作为未标注集。对于这些样本，其真实结果 Yunlab​ 将被视为未知（用于PPI方法），但这些真实值可以保留下来，用于后续评估不同推断方法的性能（例如，计算真实参数值以比较估计偏差）。Nunlab​ 将远大于 nlab​，例如 Nunlab​ 可以是 nlab​ 的5到20倍。  
4. **测试集 (Xtest​,Ytest​)**：如果需要评估 f(X) 的预测性能，或者对最终推断模型的泛化能力进行某种形式的评估，可以从剩余数据中划分一个测试集。对于本项目主要关注统计推断而言，此部分可能不是核心，但可备用。

具体的 nlab​ 和 Nunlab​ 的大小将根据ADNI数据集的实际可用样本量和特征维度来确定，并会在报告中明确说明。

#### **3.1.3. 预处理步骤**

对选定的ADNI数据将执行以下预处理步骤：

1. **数据清洗**：  
   * 处理缺失值：对于特征 X 中的缺失值，将根据缺失比例和变量类型选择合适的填补策略，例如均值/中位数填补（对于数值型特征）、众数填补（对于分类型特征），或者使用更复杂的插补方法如k-近邻插补或多重插补。如果某个特征缺失比例过高（例如 \> 40-50%），可能会考虑将其从分析中移除。目标变量 Y 在标注集中不允许有缺失。91中提到了处理缺失值的一些常见方法。  
   * 异常值处理：通过可视化（如箱线图）和统计方法（如IQR法则）识别潜在异常值，并根据其性质决定是修正、移除还是保留。  
2. **特征工程**：  
   * 根据对阿尔茨海默病领域的理解，可能会创建一些交互项或对现有特征进行转换（如对数变换、多项式扩展）以更好地捕捉 X 与 Y 之间的关系。  
   * 对于分类变量，将进行独热编码（One-Hot Encoding）或标签编码（Label Encoding），具体取决于所选ML模型的类型。  
3. **特征缩放**：对于数值型特征，将采用标准化（Standardization，使其均值为0，标准差为1）或归一化（Normalization，将其缩放到区间），以确保不同尺度的特征在ML模型训练中得到公平对待。

#### **3.1.4. 表1: 数据集特征描述**

为了清晰地呈现所用数据的概况，将包含以下表格：

**表1: ADNI数据集用于预测驱动推断的特征描述**

| 特征 | 描述 | 类型 | 角色 | 备注 (例如，单位，范围) |
| :---- | :---- | :---- | :---- | :---- |
| **总样本量 (ADNI子集)** | Ntotal​ (例如, 2000\) | \- | \- | 经过初步筛选后的样本量 |
| **标注集大小 (nlab​)** | 例如, 200 | \- | \- | 用于校正和经典推断 |
| **未标注集大小 (Nunlab​)** | 例如, 1000 | \- | \- | 用于利用ML预测增强推断 |
| **(可选)预训练/验证集大小 (Npretrain​)** | 例如, 800 | \- | \- | 若需自行训练f(X) |
| **目标变量 (Y)** | 例如, ADAS-Cog评分在12个月的变化量 | 连续型 | 结果变量 | 越高表示认知恶化越严重 |
| *Estimand of Interest* | 例如, E\[Y\] (总体平均认知变化) 或特定协变量的回归系数 βk​ | \- | \- | 本研究的核心推断目标 |
| **特征变量 (X)** |  |  |  |  |
| AGE | 基线年龄 | 连续型 | 协变量 | 岁 |
| PTEDUCAT | 教育年限 | 连续型 | 协变量 | 年 |
| APOE4 | APOE4等位基因数量 | 分类型 | 协变量 | 0, 1, 或 2 |
| MMSE | 基线简易精神状态检查评分 | 连续型 | 协变量 | 0-30分，越高表示认知功能越好 |
| LDELTOTAL | AVLT延迟回忆总分 (示例神经心理测试) | 连续型 | 协变量 | 分数 |
| Hippocampus Volume | 基线海马体积 (示例影像标记物) | 连续型 | 协变量 | mm3 |
| ... (其他相关协变量) | ... | ... | ... | ... |
| **数据来源** | Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu) | \- | \- | ADNI1, ADNI2, ADNI GO, ADNI3 等版本 |

此表格的构建对于确保研究的可复现性和透明度至关重要。它清晰地界定了研究中使用的数据规模、变量构成以及核心的推断目标，为后续的方法实施和结果解读提供了坚实的基础。通过明确标注集与未标注集的规模对比，也直观地展现了研究试图解决的“小标注数据”挑战。

### **3.2. 预训练模型 f(X) 规格**

#### **3.2.1. 模型选择**

为了从特征 X 预测结果 Y，本项目将选择一个在 scikit-learn 库中易于实现且性能稳健的机器学习模型作为预训练模型 f(X)。考虑到ADNI数据集的特点（可能包含数值和分类特征，以及中等规模的样本量），**梯度提升回归/分类器（Gradient Boosting Regressor/Classifier）** 是一个合适的选择。梯度提升机以其高预测精度、对不同类型数据的良好适应性以及对特征缩放不敏感等优点而著称。

如果目标变量 Y 是连续的（例如ADAS-Cog评分的变化量），将使用 GradientBoostingRegressor。如果 Y 是分类的（例如疾病状态的转换），将使用 GradientBoostingClassifier。

#### **3.2.2. 预测值 Y^ 的生成**

预训练模型 f(X) 将用于生成两组预测值：

1. Y^lab​=f(Xlab​)：对标注集中的样本进行预测。这些预测值将与真实标签 Ylab​ 一同用于计算PPI++方法中的校正项。  
2. Y^unlab​=f(Xunlab​)：对大规模未标注集中的样本进行预测。这些预测值将被用于构建PPI++方法中基于大量（但可能存在偏差的）信息的初步估计。

模型“预训练”过程：  
由于本项目强调利用“预训练”模型，理想情况下应使用一个在与当前研究的标注集 (Xlab​,Ylab​) 和未标注集 Xunlab​ 完全不相交的数据上训练好的模型。如果无法获取到这样一个现成的、针对ADNI数据和特定 Y 的高质量预训练模型，则将在项目内部模拟“预训练”过程。具体做法是：

* 从ADNI的原始数据中，在划分出 Xlab​ 和 Xunlab​ *之前*，预留一部分数据（即前述的 Npretrain​ 样本）专门用于训练和调优这个梯度提升模型 f(X)。  
* 模型训练将采用标准的机器学习流程，包括超参数调优（例如，使用网格搜索GridSearchCV或随机搜索RandomizedSearchCV配合交叉验证）以优化其预测性能（例如，均方误差MSE或准确率Accuracy）。  
* 一旦模型 f(X) 训练完成，其参数将被固定。然后，这个固定的模型将被用于为 Xlab​ 和 Xunlab​ 生成预测值。

这种内部“预训练”方式虽然不是严格意义上的使用外部预训练模型，但在项目范围内模拟了拥有一个固定预测模型的场景，这符合PPI框架不假设模型如何获得的要求。关键在于确保用于训练 f(X) 的数据与用于PPI校正和推断的数据之间没有重叠。

#### **3.2.3. 模拟预测模型的不完美性**

在真实世界应用中，任何ML模型的预测都不可能是完美的，它们总是带有一定的误差和潜在的偏倚。梯度提升模型本身在拟合复杂数据时就可能产生这些不完美性。为了更可控地研究PPI方法对模型不完美性的鲁棒性，或者在纯粹的模拟研究（非直接使用ADNI数据）中，可以按如下方式构造一个不完美的预测函数 f(X)：

1. 基于真实模型的扰动：  
   假设存在一个真实的潜在关系 Ytrue​=g(X)+ϵtrue​。可以先用一部分数据（或全部可用标注数据，如果是在一个纯模拟设定中）拟合一个“最优”模型 g∗(X) 来近似 g(X)。然后，不完美的预测模型 f(X) 可以通过以下几种方式构造：  
   * **添加系统偏差（Bias）**：$\\hat{Y} \= f(X) \= g^\*(X) \+ \\text{bias\_term}$，其中 $\\text{bias\_term}$ 可以是一个常数，或者是一个与 X 相关的函数，模拟模型在特定区域的系统性高估或低估。  
   * **增加随机噪声（Variance）**：$\\hat{Y} \= f(X) \= g^\*(X) \+ \\text{noise\_term}$，其中 $\\text{noise\_term}$ 是一个随机变量，其方差可以控制预测的不稳定程度。例如，$\\text{noise\_term} \\sim N(0, \\sigma^2\_{pred\\\_error})$。  
   * **使用欠拟合或过拟合模型**：故意使用一个过于简单（例如，仅包含部分重要特征的线性模型）或过于复杂（例如，未加正则化的深度网络或参数过多的树模型）的模型作为 f(X)，使其产生系统性的预测误差 27。  
   * **基于不完整或有偏特征训练**：在训练 f(X) 时，只使用 X 的一个子集，或者使用经过某种方式扭曲的特征，使得 f(X) 对 Y 的预测能力受限或带有偏见。  
2. 模拟特定错误率/准确率：  
   如果 Y 是分类变量，可以设定 f(X) 以一定的概率输出错误的类别标签，或者使其在特定类别上的表现较差，从而模拟不同水平的预测准确率和类别不平衡下的偏倚 17。例如，可以设计 f(X) 使得其对少数群体的预测准确率低于多数群体。  
3. 利用文献中关于“不完美代理标签”的讨论：  
   Egami等人 17 的工作讨论了使用大型语言模型（LLM）等产生的“不完美代理标签”进行下游统计分析的问题。他们指出，即使代理标签的准确率达到80-90%，直接使用它们也会导致显著偏差和无效的置信区间。在模拟研究中，可以借鉴这种思路，设定 f(X) 的预测准确率在一个现实的范围内（例如70%-95%），并引入一些系统性的偏差模式（例如，对某些子群体的预测更差）。

对于本项目，如果采用ADNI真实数据和内部“预训练”的梯度提升模型，其不完美性是自然产生的。如果为了更精细地控制和研究PPI++的性能，可以考虑在生成 Y^unlab​ 时，对梯度提升模型的原始预测额外叠加一层可控的偏差或噪声，从而评估PPI++在不同“质量”的预训练模型下的表现。这种对预测质量的控制，虽然增加了实验复杂度，但能更深刻地揭示PPI++方法（特别是其 λ 参数的自适应性）的优势和适用边界。

### **3.3. 推断方法的实现 (Python 3.12)**

所有推断方法都将使用Python 3.12版本进行实现。核心库包括 numpy 用于数值计算，pandas 用于数据管理，scikit-learn 用于机器学习模型的辅助（如数据划分、指标计算），statsmodels 用于实现经典统计模型，以及专门的 ppi\_py 库用于预测驱动推断。

#### **3.3.1. 方法1：预测驱动推断 (PPI++)**

* **库的使用**：将直接利用 aangelopoulos/ppi\_py 这个专门为预测驱动推断开发的Python库 29。该库实现了包括PPI++在内的多种PPI方法。  
* **函数调用**：根据需要推断的参数类型，选择 ppi\_py 中相应的函数。例如：  
  * 估计总体均值 E\[Y\]：使用 ppi\_mean\_ci(Y\_lab, Yhat\_lab, Yhat\_unlab, alpha=0.05) 来获取置信区间，以及 ppi\_mean\_pointestimate 获取点估计 22。如果需要p值，可以使用 ppi\_mean\_pval。  
  * 估计总体分位数：使用 ppi\_quantile\_ci 和 ppi\_quantile\_pointestimate。  
  * 估计OLS回归系数：使用 ppi\_ols\_ci 和 ppi\_ols\_pointestimate。此时，输入除了 Ylab​,Y^lab​,Y^unlab​ 外，还需要相应的协变量矩阵 Xlab​ 和 Xunlab​。  
  * 估计Logistic回归系数：使用 ppi\_logistic\_ci 和 ppi\_logistic\_pointestimate。  
* **参数设置**：  
  * Y\_lab：标注集中的真实标签。  
  * Yhat\_lab：预训练模型 f(X) 对标注集 Xlab​ 的预测 Y^lab​。  
  * Yhat\_unlab：预训练模型 f(X) 对未标注集 Xunlab​ 的预测 Y^unlab​。  
  * alpha：显著性水平，通常设为0.05，以构建95%的置信区间。  
  * lam (功效调整参数)：ppi\_py 库中的PPI++实现允许 lam 参数自动从数据中估计，这是其默认行为 22。这将是本项目的首选方式，以充分利用PPI++的自适应性。如果需要，也可以手动设置 lam（例如，lam=0 退化到经典推断，lam=1 对应无功效调整的PPI）。  
  * coord (坐标)：在估计多维参数（如回归系数向量）时，lam 的优化可以针对所有坐标的总方差，或者针对某个特定坐标的方差 22。默认情况下，ppi\_py 会优化总方差。

#### **3.3.2. 方法2：基准线 \- 直接ML预测为基础的推断（朴素推断）**

这种方法的核心思想是将预训练模型 f(X) 在未标注数据上产生的预测 Y^unlab​ 视为这些样本的“真实”结果，然后将它们与已有的标注数据 (Xlab​,Ylab​) 合并，形成一个“增强的”数据集。之后，在这个增强数据集上应用经典的统计推断方法。

* **构建增强数据集**：  
  * 特征矩阵 Xaug​=concatenate(Xlab​,Xunlab​)  
  * 结果向量 Yaug​=concatenate(Ylab​,Y^unlab​)  
  * 总样本量 Naug​=nlab​+Nunlab​  
* **应用经典统计方法**：  
  * **估计总体均值 E\[Y\]**：计算 Yaug​ 的样本均值 Yˉaug​=Naug​1​∑Yaug,i​。其置信区间将基于中心极限定理，使用 Yaug​ 的样本标准差 saug​ 计算：Yˉaug​±z1−α/2​Naug​​saug​​。  
  * **估计OLS回归系数**：在 (Xaug​,Yaug​) 上拟合一个普通最小二乘（OLS）回归模型。例如，使用 statsmodels.api.OLS(Y\_aug, sm.add\_constant(X\_aug)).fit()。然后从拟合结果中提取系数估计值及其标准误，并构建置信区间。

这种方法本质上是一种朴素的插补（imputation）策略，它忽略了 Y^unlab​ 中固有的预测误差和潜在偏差。正如文献 15 所警告的，“将这些ML预测直接代入而不考虑预测误差，会使回归分析产生偏差、不一致性，并且过度自信（即标准误过小）”。31 和 32 也将此类方法称为“朴素估计器”（Naive Estimator），并指出其可能导致有偏的推断。因此，预期这种基准方法的置信区间覆盖率可能会不达标。

#### **3.3.3. 方法3：基准线 \- 经典推断（仅使用标注数据）**

这种方法代表了在没有未标注数据或不信任任何ML预测时的标准做法。所有统计推断仅基于小规模的标注数据集 (Xlab​,Ylab​)。

* **应用经典统计方法**：  
  * **估计总体均值 E\[Y\]**：计算 Ylab​ 的样本均值 Yˉlab​=nlab​1​∑Ylab,i​。其置信区间将基于t分布（因为 nlab​ 较小）：Yˉlab​±tnlab​−1,1−α/2​nlab​​slab​​，其中 slab​ 是 Ylab​ 的样本标准差。  
  * **估计OLS回归系数**：在 (Xlab​,Ylab​) 上拟合OLS回归模型，例如使用 statsmodels.api.OLS(Y\_lab, sm.add\_constant(X\_lab)).fit()。提取系数估计值和标准误，构建置信区间。

由于 nlab​ 较小，预期这种方法的置信区间虽然在理论上是有效的（假设经典统计模型的假设成立），但其长度可能会比较宽，反映了基于小样本推断的不确定性较大 7。

### **3.4. 开发环境与工具**

* **Cursor AI 编辑器**：整个项目的代码编写、调试、文档撰写（包括本报告的Markdown和LaTeX版本）将主要在Cursor AI编辑器中完成 33。  
  * **代码辅助**：利用其AI驱动的代码生成、自动补全、智能重写和错误修正功能，提高Python脚本的开发效率和质量 34。  
  * **上下文理解与问答**：通过 @ 符号引用文件、代码片段或文档，利用其聊天功能查询代码库、理解复杂逻辑或获取API用法建议 33。  
  * **Jupyter Notebook风格开发（如果需要）**：Cursor支持在 .py 文件中使用 \# %% 单元格分隔符进行类似Jupyter Notebook的交互式开发和探索，这对于数据预处理和初步分析阶段可能很有用 37。  
  * **报告撰写**：利用Cursor对Markdown和LaTeX语法的支持，以及可能的AI辅助写作功能，撰写项目报告 39。  
* **Python 核心库**：  
  * numpy 30: 用于高效的数值数组操作和数学函数。  
  * pandas: 用于数据结构（如DataFrame）的创建、管理和预处理 41。  
  * scikit-learn: 用于实现梯度提升模型（如果需要内部“预训练”）、数据划分（如train\_test\_split）、性能度量（如MSE、准确率）以及可能的预处理步骤（如StandardScaler）41。  
  * ppi\_py: Angelopoulos等人开发的用于预测驱动推断的专用库，将是实现PPI++方法的核心 29。  
  * statsmodels: 用于实现经典的OLS回归和Logistic回归模型，并获取其详细的统计推断结果（系数、标准误、置信区间、p值）41。  
  * matplotlib 和 seaborn: 用于生成结果可视化图表，如箱线图、置信区间比较图等 41。  
* **版本控制**：使用Git进行代码和报告的版本管理，并托管在如GitHub的平台上，以确保项目的可追溯性和协作潜力 43。Cursor AI编辑器通常与Git有良好的集成 33。  
* **报告生成工具 (Quarto)**：考虑到需要同时生成Markdown和高质量的PDF (通过LaTeX) 报告，并且项目以Python为核心，强烈推荐使用Quarto 44。Quarto是R Markdown的下一代产品，对Python有优秀的支持，并且可以很好地与VS Code (Cursor AI基于此) 集成。  
  * **单一源文件**：使用Quarto的 .qmd 格式可以从单一源文件生成多种输出格式。  
  * **代码块执行**：直接在文档中嵌入和执行Python代码块，并将结果（如表格、图形）无缝包含在报告中。  
  * **数学公式**：支持LaTeX语法渲染数学公式 46。  
  * **表格**：可以将pandas DataFrame通过 tabulate 库 47 或新兴的 Great Tables 库 48 转换为Markdown或直接输出为LaTeX表格，以获得出版质量。Quarto的 output: asis 选项对直接输出LaTeX代码块非常有用 48。  
  * **图形**：Matplotlib和Seaborn生成的图形可以轻松嵌入，Quarto还支持子图、图注和交叉引用 50。  
  * **交叉引用**：Quarto提供强大的交叉引用功能，可以引用章节、图、表、公式等 51。  
  * **文献管理**：通过 .bib 文件和CSL样式管理参考文献 54。

采用Quarto将极大地简化报告撰写和格式转换的流程，确保内容的一致性和专业性，同时也符合可复现研究的最佳实践。

一个需要特别注意的实施细节是基准方法2（直接ML预测为基础的推断）的操作化。它不仅仅是获得ML模型的点预测值。为了进行“推断”，这意味着需要将这些预测值（Y^unlab​）视为真实观察值，然后与标注数据 (Xlab​,Ylab​) 合并。之后，对这个混合了真实值和预测值的“增强”数据集应用标准的统计程序（例如，对 (X,Y^) 进行OLS回归以获得系数的置信区间）。这个过程必须清晰地阐述其（通常被违反的）假设，例如假设 Y^unlab​ 与真实的 Yunlab​ 具有相同的统计特性，或者误差项满足经典模型的假定。实验结果预期会暴露这种朴素方法的缺陷，例如置信区间覆盖率不足或标准误估计不当，从而凸显PPI++等经过校正的方法的优越性。整个实验设计和实施过程将严格遵循可复现研究的原则 43，所有代码、数据处理步骤和库版本都将被详细记录和公开。

## **4\. 结果与评估**

本章节将详细呈现对比实验所获得的各项发现。评估的焦点在于比较三种推断方法——预测驱动推断（PPI++）、基于直接ML预测的朴素推断、以及仅使用标注数据的经典推断——在估计目标参数时的性能。结果的组织将遵循模拟研究报告的最佳实践 57。

### **4.1. 评估指标**

为了全面比较三种推断方法的性能，将采用以下标准统计指标：

1. **点估计量性能 (例如，对总体均值 E\[Y\] 或回归系数 β 的估计 θ^)**:  
   * **偏差 (Bias)**：衡量估计量平均偏离真实参数值的程度。如果真实参数值 θ 已知（例如，在模拟子研究中，或者使用完整数据集计算得到的参数作为“真实”代理），偏差定义为 Bias(θ^)=E\[θ^\]−θ。在单次实验中，如果真实值未知，则主要关注估计的稳定性和与其他方法的一致性，但偏差的直接量化可能困难。如果通过多次重复实验（例如，对标注集和未标注集进行多次随机抽样和划分）来模拟，则可以计算平均偏差 58。  
   * **均方误差 (Mean Squared Error, MSE)**：衡量估计量与真实参数值之间平均平方差异，综合了偏差和方差的影响。MSE(θ^)=E\[(θ^−θ)2\]=Var(θ^)+(Bias(θ^))2。与偏差类似，MSE的精确计算也依赖于真实参数值或重复实验 58。  
2. **置信区间 (CI) 性能**:  
   * **经验覆盖概率 (Empirical Coverage Probability, ECP)**：在多次重复实验中，所构建的 (1−α) 置信区间包含真实参数值的比例。理想情况下，ECP应接近于名义覆盖水平 (1−α)（例如95%）58。这是衡量置信区间有效性的核心指标。  
   * **平均长度 (Average Length)**：置信区间的平均宽度。在保证ECP达到名义水平的前提下，置信区间越短，表示推断越精确 58。  
3. **假设检验性能 (如果适用，例如 ppi\_py 提供了 ppi\_mean\_pval 用于均值检验)**:  
   * **第一类错误率 (Type I Error Rate)**：当原假设为真时，错误地拒绝原假设的概率。在多次重复实验中，这应接近预设的显著性水平 α 58。  
   * **统计功效 (Power)**：当备择假设为真（即原假设为假）时，正确地拒绝原假设的概率。功效越高，方法检测真实效应的能力越强 58。

由于本项目主要基于单个（真实）数据集的划分进行一次性比较，而非大规模蒙特卡洛模拟研究，因此对于偏差、MSE、ECP和第一类错误率/功效的精确经验估计可能受限。在这种情况下，评估将更侧重于：

* 比较不同方法产生的点估计值之间的差异，并结合领域知识判断其合理性。  
* 比较置信区间的长度，并讨论其覆盖真实参数（如果可以通过某种方式合理估计）的可能性。  
* 如果可能，通过对标注集进行自助法（bootstrap）重采样来近似评估估计量和置信区间的稳定性，但这并非PPI的核心评估方式。

核心比较将围绕置信区间的长度和（理论上的）有效性展开，因为PPI方法的主要承诺是提供有效的、且通常更窄的置信区间。

### **4.2. 比较性能分析**

将针对所选的ADNI数据集和具体推断目标（例如，估计APOE4基因型对某项认知指标的平均影响，控制年龄和教育程度后的回归系数），系统地展示三种方法的推断结果。

#### **表2: 点估计量比较性能 (示例结构)**

如果进行重复采样实验，此表将更有意义。在单次实验中，可以展示点估计值本身，并辅以讨论。

| 目标参数 (Estimand) | 推断方法 | 点估计值 (θ^) | (如果模拟) 偏差 (Bias) | (如果模拟) 均方误差 (MSE) | (如果模拟) MCSE (Bias) | (如果模拟) MCSE (MSE) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **总体均值 E\[Y\]** (例如, 1年MMSE变化量) | PPI++ | μ^​PPI++​ | ... | ... | ... | ... |
|  | 直接ML预测推断 (朴素) | μ^​Naive​ | ... | ... | ... | ... |
|  | 经典推断 (仅标注数据) | μ^​Classic​ | ... | ... | ... | ... |
| **回归系数 βAGE​** (Y 对年龄) | PPI++ (OLS) | β^​AGE,PPI++​ | ... | ... | ... | ... |
|  | 直接ML预测推断 (朴素 OLS) | β^​AGE,Naive​ | ... | ... | ... | ... |
|  | 经典推断 (仅标注数据 OLS) | β^​AGE,Classic​ | ... | ... | ... | ... |
| ... (其他目标参数) | ... | ... | ... | ... | ... | ... |

*注：MCSE 指蒙特卡洛标准误，仅在进行多次重复模拟实验时适用。对于基于单一真实数据集的分析，偏差和MSE的直接计算需要一个“真实”参数值，这可能通过使用整个数据集（如果足够大且被认为是总体）计算得到，或者通过领域知识设定。*

此表旨在直接比较不同方法产生的参数估计值的准确性和精度。预期PPI++方法在偏差和MSE方面（尤其是在重复实验的平均意义上）会优于或至少不差于经典方法，并显著优于可能存在严重偏差的朴素ML预测推断。

#### **表3: 置信区间性能比较 (示例结构)**

这是本研究的核心成果表格，直接反映统计推断的有效性和效率。

| 目标参数 (Estimand) | 推断方法 | (如果模拟) 经验覆盖率 (ECP) | 置信区间长度 (Avg. Length) | 95% 置信区间示例 | (如果模拟) MCSE (ECP) | (如果模拟) MCSE (Length) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **总体均值 E\[Y\]** (例如, 1年MMSE变化量) | PPI++ | (目标: 0.95) | LPPI++​ | \[lower, upper\]PPI++​ | ... | ... |
|  | 直接ML预测推断 (朴素) | (可能 \< 0.95) | LNaive​ | \[lower, upper\]Naive​ | ... | ... |
|  | 经典推断 (仅标注数据) | (目标: 0.95) | LClassic​ | \[lower, upper\]Classic​ | ... | ... |
| **回归系数 βAGE​** (Y 对年龄) | PPI++ (OLS) | (目标: 0.95) | LPPI++′​ | \[lower, upper\]PPI++′​ | ... | ... |
|  | 直接ML预测推断 (朴素 OLS) | (可能 \< 0.95) | LNaive′​ | \[lower, upper\]Naive′​ | ... | ... |
|  | 经典推断 (仅标注数据 OLS) | (目标: 0.95) | LClassic′​ | \[lower, upper\]Classic′​ | ... | ... |
| ... (其他目标参数) | ... | ... | ... | ... | ... | ... |

此表将清晰地展示PPI++方法是否能够提供具有名义覆盖率（例如95%）的置信区间，并且其平均长度是否比仅使用标注数据的经典方法更短。同时，它也将揭示直接使用ML预测进行推断可能导致的覆盖率不足问题 1。PPI方法的核心优势正在于此：在保证统计有效性的前提下，通过利用更多信息（未标注数据和ML预测）来提高推断的精度。

#### **4.2.3. 可视化结果**

为了更直观地展示结果，将采用以下可视化方法：

* **点估计与置信区间图**：对于关键的1-2个目标参数，绘制一个比较图，显示三种方法得到的点估计值以及相应的95%置信区间。这将直观地展示区间长度和估计值位置的差异。  
* **(如果进行重复实验) 覆盖率与区间长度散点图**：如果实验涉及多次重复（例如，通过对标注/未标注数据的不同随机抽样子集进行分析），可以绘制散点图，其中每个点代表一次实验运行，横轴为置信区间长度，纵轴为是否覆盖真实参数（0或1）。然后可以比较不同方法下点的分布情况，以及计算出的平均覆盖率和平均长度。  
* **(如果适用) 校正项或 λ 参数的分布**：如果PPI++中的校正项或功效调整参数 λ 具有有趣的分布或取值，可以考虑将其可视化（例如，直方图或箱线图），以洞察模型是如何利用预测信息的。

#### **预期结果的理论支撑**

基于预测驱动推断的理论 1 和PPI++的特性 19，可以预期以下结果模式：

1. **PPI++方法**：  
   * **置信区间覆盖率**：应接近或达到预设的名义水平（例如95%），表明其推断的统计有效性。  
   * **置信区间长度**：相比于仅使用标注数据的经典推断方法，PPI++的置信区间长度应显著更短，尤其是在预训练模型 f(X) 具有一定预测能力时。这反映了其利用大量未标注数据和ML预测所带来的效率提升。  
   * **点估计**：其点估计量应是（渐进）无偏的。  
2. **直接ML预测为基础的推断（朴素推断）**：  
   * **置信区间覆盖率**：可能远低于名义水平。这是因为该方法忽略了ML预测中的误差和偏差，导致标准误被低估，置信区间构建不当 15。  
   * **置信区间长度**：可能表现出误导性的窄，给人以高精度的假象，但这种精度是以牺牲有效性为代价的。  
   * **点估计**：可能存在显著偏差，偏差的方向和大小取决于ML模型的具体偏误。  
3. **经典推断（仅使用标注数据）**：  
   * **置信区间覆盖率**：在经典统计模型的假设（如正态性、同方差性等，如果适用）得到满足的前提下，其覆盖率应接近名义水平。  
   * **置信区间长度**：由于仅使用了少量标注数据，其置信区间长度预计会是三者中最宽的，反映了信息量的不足 7。  
   * **点估计**：在经典假设下是无偏的，但方差较大。

实验结果的核心在于验证PPI++是否能在实践中兑现其理论承诺：即在保持统计有效性（正确的覆盖率）的同时，通过融合ML预测和未标注数据，显著优于仅依赖小样本标注数据的经典方法，并纠正朴素使用ML预测所带来的统计谬误。此外，PPI++中自适应调整的 λ 参数如何反映预训练模型 f(X) 的质量，也是一个值得关注的方面。如果 f(X) 预测能力强，λ 应该较大，使得PPI++能更充分地从预测中获益；反之，如果 f(X) 预测能力弱，λ 应该较小，使PPI++的表现更接近于稳健的经典方法，这体现了其“自适应”和“稳健”的特性。

## **5\. 讨论**

### **5.1. 结果解读**

在对ADNI数据集应用三种不同的统计推断方法后，实验结果（如表2和表3所示）为我们提供了关于预测驱动推断（具体为PPI++）在特定情景下性能的宝贵信息。

首先，关于**点估计的性能**，如果进行了重复实验或有可靠的真实参数值进行比较，我们预期PPI++方法产生的点估计量（例如，总体均值或回归系数）将展现出较低的偏差和均方误差。这归因于PPI++框架利用了标注数据对ML预测的偏差进行校正，并结合了来自大量未标注数据的信息。相比之下，直接使用ML预测进行推断（朴素方法）可能由于ML模型固有的系统性偏差而导致点估计值偏离真实参数较远。而仅使用少量标注数据的经典方法，虽然其点估计在理论上是无偏的（在满足其模型假设的前提下），但由于样本量小，其方差可能较大，从而导致单次实验中的估计值波动较大，均方误差也可能较高。

其次，也是更为关键的，关于**置信区间的性能**，结果清晰地展示了不同方法的权衡。

* **PPI++方法**：预期其构建的置信区间能够达到或非常接近预设的名义覆盖率（例如95%）。这是PPI框架的核心保证，即在利用ML预测的同时维持统计推断的有效性 1。更重要的是，与经典方法相比，PPI++的置信区间长度通常会显著缩短。这种长度的缩减直接反映了统计效率的提升，意味着在相同的置信水平下，我们对参数的估计更为精确。这种效率提升来源于对大量未标注数据中预测信息的有效利用。  
* **直接ML预测推断（朴素方法）**：如理论所预警 15，这种方法构建的置信区间很可能无法达到名义覆盖率。其原因在于它未能校正ML预测的偏差，并且可能低估了由预测误差引入的额外变异性，导致标准误计算不准确，区间过于“自信”而偏窄。  
* **经典推断（仅标注数据）**：这种方法在满足其统计假设时，通常能保证置信区间的有效覆盖率。然而，由于其仅依赖于少量标注数据，其置信区间长度预计会是三者中最宽的，这反映了小样本推断固有的不确定性较大、精度较低的问题 7。

如果实验结果与这些预期一致，则有力地证明了PPI++在当前设定下，确实能够实现“两全其美”：既保证了统计推断的有效性（不像朴素ML方法那样可能失效），又通过整合更多信息源提升了推断的效率（比经典小样本方法更精确）。

任何与预期不符的意外结果都值得深入分析。例如，如果PPI++的置信区间长度并未显著优于经典方法，可能的原因包括：(1) 所选用的预训练ML模型 f(X) 预测能力非常差，以至于其提供的预测信息几乎没有价值（此时PPI++的 λ 参数可能会自适应地趋近于0）；(2) 数据集本身的特性使得 X 与 Y 之间的关系非常弱或噪声极大，即使有大量未标注数据也难以提取有效信息；(3) nlab​ 相对于 Nunlab​ 的比例或绝对数量对于当前问题而言仍然不足以精确估计校正项。

### **5.2. 预训练模型质量的影响**

预训练机器学习模型 f(X) 的质量对预测驱动推断的性能，特别是效率提升幅度，具有直接影响。虽然PPI/PPI++框架在理论上对 f(X) 的质量不作假设，依然能保证推断的有效性（即置信区间的覆盖率）1，但 f(X) 的预测准确性越高，PPI++方法从中获益就越大，表现为置信区间长度的缩减越显著 1。

在本项目中，如果使用了ppi\_py库并让其自动估计PPI++的功效调整参数 λ，那么 λ 的取值本身就间接反映了模型 f(X) 所提供预测的相对效用。

* 如果估计得到的 λ 值接近1，表明模型 f(X) 的预测与真实标签之间的（经过校正后的）关联性较强，PPI++能够充分信任并利用这些预测来减小估计量的方差。  
* 如果估计得到的 λ 值接近0，则表明模型 f(X) 的预测质量较低，或者其预测带来的信息增益不足以抵消引入的噪声，此时PPI++会自动降低对预测的依赖，其表现将趋向于仅使用标注数据的经典推断方法。  
* λ 值介于0和1之间，则表示PPI++在经典估计和完全依赖（校正后）预测的估计之间进行了一种数据驱动的优化权衡。

这种自适应性是PPI++的核心优势之一 19，它使得该方法在面对质量未知的“黑箱”预训练模型时依然表现稳健。在讨论部分，可以结合所用 f(X) 模型（例如梯度提升机）在某个独立验证集上的预测性能指标（如R2、MSE或AUC），与最终PPI++中估计出的 λ 值进行关联分析，探讨预测性能与 λ 取值以及最终推断效率提升之间是否存在预期的联系。例如，如果梯度提升模型本身的预测能力一般，那么即使PPI++仍然有效，其置信区间的缩短幅度可能不会像使用一个高度精确的 f(X) 模型那样惊人。

### **5.3. 研究局限性**

本研究虽然力求严谨，但也存在一些固有的局限性：

1. **单一数据集和特定任务**：本研究主要基于ADNI数据集和特定的推断目标（例如，某个回归系数或均值的估计）。研究结果的普适性可能受到所选数据集特性（如变量间的关系复杂度、信噪比、样本分布等）和任务类型的影响。在其他类型的数据集或不同的推断问题上，三种方法的相对性能可能会有所不同。  
2. **预训练模型的选择与“预训练”方式**：如果预训练模型 f(X) 是在项目内部通过划分一部分ADNI数据进行“预训练”的，那么这个 f(X) 的质量和特性可能与外部获取的、在更大数据集上训练的真正“预训练”模型有所差异。此外，所选的梯度提升模型只是众多ML模型中的一种，使用其他类型的模型（如深度神经网络）可能会产生不同的预测质量，进而影响PPI++的效率增益。  
3. **对分布偏移的简化处理**：虽然PPI框架理论上可以处理协变量偏移和标签偏移 1，但本项目可能未对此进行深入的实证研究，除非所选的ADNI数据子集明确设计用来体现这种偏移。在实际应用中，分布偏移是一个常见且复杂的问题，其对PPI性能的影响值得进一步探讨。  
4. **小样本推断的固有挑战**：尽管PPI++旨在提升小标注样本下的推断效率，但当标注样本量 nlab​ 极小时，校正项的估计本身也可能存在较大不确定性，这可能会限制PPI++相对于经典方法的优势。经典统计方法在小样本下的一些固有假设（如正态性假设对于t检验的有效性）的满足程度也会影响比较的基准。  
5. **计算资源和时间限制**：全面的蒙特卡洛模拟研究（涉及大量重复实验以精确评估覆盖率、偏差等）通常需要大量的计算资源和时间。本入门级项目可能主要依赖于单次或少数几次数据划分下的比较，其结论的统计稳定性可能不如大规模模拟研究。

### **5.4. 实际意义与建议**

尽管存在上述局限性，本研究的发现对于数据科学实践仍具有重要的指导意义：

1. **在标注数据稀缺时增强推断能力**：当面临高质量标注数据获取困难，但拥有大量未标注数据和（或）一个可用的预训练ML模型时，预测驱动推断（如PPI++）提供了一种统计上有效且通常更高效的进行参数估计和假设检验的途径。它使得研究者能够“榨取”未标注数据和ML预测中的信息价值，而不仅仅是依赖有限的标注样本。  
2. **避免朴素使用ML预测的陷阱**：本研究应能清晰地警示直接将ML预测结果等同于真实值并用于下游统计分析的风险。PPI++通过其校正机制，为如何负责任地使用这些不完美的预测提供了具体方法。  
3. **对ML模型选择的启示**：虽然PPI对ML模型本身不作假设，但一个预测能力更强的ML模型通常会带来更大的统计效率提升。因此，在资源允许的情况下，选择或构建一个尽可能准确的预测模型 f(X) 是有益的。然而，PPI++的自适应性也意味着即使只有一个中等质量的预测模型，该方法仍然可能优于经典的小样本推断。  
4. **适用场景**：预测驱动推断特别适用于那些传统上依赖于昂贵实验或人工标注的领域，例如生物医学研究（如本例中的ADNI数据分析）、药物研发 19、遥感图像分析、社会科学调查数据分析 17 等。在这些领域，利用PPI可以潜在地减少对大规模标注的需求，加速研究进程，或在现有数据条件下获得更精确的结论。

建议在实际应用中，如果条件满足（存在少量标注数据、大量未标注数据以及一个预测模型），应优先考虑使用如PPI++这样的预测驱动推断方法，而非仅依赖小样本经典推断或冒险使用未经校正的ML预测进行推断。

### **5.5. 未来工作**

基于本研究的发现和局限性，未来可以从以下几个方向进行扩展：

1. **探索其他PPI变体**：比较PPI++与其他先进的PPI方法，如交叉预测推断（Cross-Prediction Inference）70（特别适用于需要在同一数据集上训练预测模型和进行推断的场景，能更有效地利用数据）、分层PPI（Stratified PPI）72（适用于数据存在异质性，ML模型在不同子群体上表现不同的情况），或基于E值的PPI 4（能处理更广泛的推断问题，如变化点检测和因果发现，并具有随时有效的特性）。  
2. **系统性评估模型质量的影响**：设计更全面的模拟研究，系统地改变预训练模型 f(X) 的预测准确率、偏差大小和类型，以量化这些因素对PPI++（特别是参数 λ 的选择和最终效率增益）的具体影响。  
3. **不同类型参数和模型的推断**：将PPI方法应用于更复杂的参数估计问题，例如非线性模型的参数、生存分析中的风险比，或广义线性模型族中的其他参数。  
4. **处理更复杂的分布偏移**：在存在显著协变量偏移或标签偏移的数据集上，实证检验和比较不同PPI调整策略的性能。  
5. **可解释性与诊断工具**：开发与PPI方法配套的可解释性工具或诊断程序，帮助用户理解预测信息是如何影响最终推断结果的，以及何时PPI可能表现不佳。  
6. **大规模真实世界应用验证**：在更多不同领域的真实大规模数据集上应用和验证PPI方法的有效性和实用性。

本研究的核心价值在于揭示了在现代数据科学环境中，机器学习的预测能力与统计推断的严谨性可以有效结合。预测驱动推断框架，特别是PPI++，通过其精巧的偏差校正和自适应机制，为从有限的标注数据和丰富的预测信息中提取可靠的统计结论提供了一条有力的途径。这不仅关乎统计方法的进步，更关乎如何在人工智能时代进行更可信、更高效的数据驱动的科学探索。例如，在药物临床试验中，若能利用疾病进展模型对安慰剂组的“数字孪生”进行预测，并通过PPI校正，则可能在保证试验有效性的前提下，减少所需招募的对照组患者数量，或在同样样本量下获得更高的检验效能 19。这种潜力对于加速科学发现和降低研究成本具有重要意义。

此外，关于预训练模型 f(X) 的训练过程，如果它必须在项目内部完成（即利用部分可用标注数据），那么数据划分的策略就变得非常关键。传统的做法是将一部分标注数据用于训练 f(X)，另一部分用于PPI的校正步骤。然而，这种数据分割可能导致用于校正的样本量进一步减少，从而影响PPI的性能。交叉预测推断（Cross-prediction）70 等方法正是为了解决这个问题而提出的，它们通过类似交叉验证的机制，允许在训练预测模型和进行推断时更有效地利用全部标注数据，避免了“次优的数据分割” 70。虽然本项目可能主要聚焦于PPI++并假设 f(X) 是给定的，但在讨论其局限性或展望未来工作时，提及这种数据利用效率的问题以及交叉预测等更高级的解决方案，能够体现对该领域更深层次的理解。

## **6\. 结论**

本项目旨在探索和评估一种在拥有少量标注数据、大量未标注数据和预训练机器学习模型的情景下进行有效统计推断的方法。通过对预测驱动推断（PPI）框架，特别是其增强版PPI++的理论学习和基于ADNI数据集的实证分析，本研究旨在验证该方法相对于传统小样本推断和朴素使用ML预测进行推断的优势。

主要目标包括：

1. 实现一个基于PPI++的统计推断流程，该流程能够整合预训练模型（例如梯度提升机）对大量未标注数据的预测，并通过少量标注数据进行偏差校正。  
2. 将PPI++方法的性能（主要体现在置信区间的有效覆盖率和长度）与两种基准方法进行比较：一是直接依赖ML模型预测值进行推断，二是仅使用少量标注数据进行经典统计推断。  
3. 利用Python 3.12和Cursor AI编辑器完成整个项目的代码实现、实验运行和报告撰写，确保研究过程的可复现性。

根据理论预期和相关文献 1，本研究的核心发现预计将表明：

* **PPI++方法** 能够提供统计上有效的置信区间（即经验覆盖率接近名义水平），同时相比仅使用少量标注数据的经典推断方法，其置信区间长度显著更短，从而提高了推断的精度和效率。这得益于其对大量未标注数据中预测信息的有效利用以及通过功效调整参数 λ 实现的对预测质量的自适应性。  
* **直接基于ML预测的朴素推断方法** 很可能由于未能充分校正ML模型的预测偏差和误差，导致其构建的置信区间覆盖率不达标，或产生具有误导性的过窄区间，从而损害统计推断的有效性。  
* **仅使用少量标注数据的经典推断方法** 虽然在满足其假设条件下能提供有效的置信区间，但由于样本量所限，其区间通常较宽，统计功效较低。

综上所述，本项目预期将证实预测驱动的统计推断方法，特别是PPI++，为在当前普遍存在标注数据稀缺而未标注数据和预训练模型丰富的场景下进行科学研究和数据分析，提供了一种兼具统计严谨性和实践高效性的强大工具。它不仅能够帮助研究者从有限的数据中提取更多有价值的信息，还有助于推动机器学习的预测能力更可靠地服务于统计推断和科学发现。最终，本研究将以一份完整的、可复现的报告形式，详细呈现所有的方法、过程、结果与讨论，为数据科学导论课程贡献一个关于现代统计推断前沿应用的实例。

## **7\. 可复现性**

为了确保本研究的透明度和可复现性，所有相关的代码、数据处理脚本、以及生成报告的源文件都将遵循以下准则进行组织和公开。

### **7.1. 代码可用性**

* 所有用于数据预处理、模型训练（如果内部“预训练”f(X)）、三种推断方法实现（PPI++、朴素ML推断、经典推断）、以及结果评估和可视化的Python脚本（.py文件）或Jupyter Notebook（.ipynb文件，如果用于探索性分析）将被妥善组织。  
* 代码将托管在一个公开的Git仓库（例如GitHub）中。仓库链接将在最终报告中提供。  
* 将明确指出所使用的核心Python库及其版本号，特别是 numpy、pandas、scikit-learn、statsmodels，以及 ppi\_py (例如，ppi\_py 版本可参考其官方发布，numpy 版本如\~1.26.1 30)。  
* 代码将包含清晰的注释，解释关键部分的逻辑和参数选择。

### **7.2. 数据可用性**

* 本研究计划使用公开的阿尔茨海默病神经影像学倡议（ADNI）数据库。将在报告中提供获取ADNI数据的官方途径（adni.loni.usc.edu）和相应的引用指南 24。  
* 由于ADNI数据的使用通常需要注册和批准，原始数据本身不会直接包含在项目仓库中。但是，所有用于从原始ADNI数据中筛选、提取和预处理研究所需特定子集（包括特征 X 和结果 Y）的脚本将被提供。  
* 数据划分（标注集、未标注集、预训练集（如果需要））的具体参数和随机种子（如果适用）将被记录，以确保他人可以重现相同的数据分割。  
* 如果生成了中间数据文件（例如，预处理后的特征矩阵），其格式和生成方式将被说明。

### **7.3. 复现说明**

* **环境设置**：将在项目仓库的README文件中提供详细的环境设置说明，包括：  
  * Python版本（3.12）。  
  * 所需Python库及其推荐版本的列表（例如，通过 requirements.txt 文件）。  
  * 安装这些库的推荐方法（例如，使用 pip install \-r requirements.txt）。  
* **运行流程**：将提供一个清晰的步骤指南，说明如何从获取（或模拟获取）数据开始，依次运行数据预处理脚本、模型训练脚本（如果适用）、各种推断方法实现脚本，并最终生成报告中的表格和图形结果。  
* **Cursor AI 编辑器**：虽然本项目使用Cursor AI编辑器进行开发，但所有核心的Python代码都应能在标准的Python环境中执行，无需Cursor AI的特定功能。报告中会提及Cursor AI在开发过程中的辅助作用（如代码生成、调试、文档编写辅助 33），但这不应成为复现研究结果的障碍。  
* **报告生成 (Quarto)**：如果使用Quarto生成最终的MD和PDF报告，将提供 .qmd 源文件以及编译报告所需的任何特定Quarto扩展或LaTeX包（如果使用了非标准包）的说明。Quarto项目本身可以很好地组织代码、文本和输出，进一步增强可复现性 43。

遵循这些可复现性措施，旨在使其他研究者或课程评估者能够理解、验证并在必要时扩展本研究的工作。这符合现代数据科学和统计研究的最佳实践 43。

## **参考文献**

* Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T. (2023). Prediction-Powered Inference. *Science, 382*(6669), 669-674. 1  
* Angelopoulos, A. N., Duchi, J. C., & Zrnic, T. (2023). PPI++: Efficient Prediction-Powered Inference. *arXiv preprint arXiv:2311.01453*. 19  
* Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Cross-prediction: A method for valid inference powered by machine learning. *Proceedings of the National Academy of Sciences, 120*(50), e2322083121. 70  
* Boyeau, P., Veyrier, C., & Guedj, B. (2025). Prediction-powered inference for clinical trials. *medRxiv*. 19  
* Candès, E. J., & Zrnic, T. (2023). Cross-prediction-powered inference. *arXiv preprint arXiv:2309.16598*. 71  
* Chatzi, M., Chen, Z., d’Alché-Buc, F., Foster, D. P., & Stine, R. (2024). Stratified Prediction-Powered Inference. *Advances in Neural Information Processing Systems (NeurIPS)*. 72  
* Csillag, D., Struchiner, C. J., & Goedert, G. T. (2025). Prediction-Powered E-Values. *arXiv preprint arXiv:2502.04294*. 4  
* Egami, N., Hinck, M., Stewart, B. M., & Wei, H. (2023). Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of Large Language Models. *arXiv preprint arXiv:2306.04746*. 17  
* GitHub Repository. aangelopoulos/ppi\_py. 29  
* GitHub Repository. yang3kc/cursor\_latex\_template. 39  
* Kallus, N. (n.d.). Research. Nathan Kallus. 79  
* Kluger, D. M., et al. (2025). Prediction-Powered Inference with Imputed Covariates and Nonuniform Sampling. *arXiv preprint arXiv:2501.18577*. 31  
* Lecun, Y., Cortes, C., & Burges, C. J. C. (n.d.). MNIST handwritten digit database..  
* Lin, W., et al. (2020). What are the limitations of classical statistical inference methods (e.g., t-tests, chi-squared tests, linear regression) when applied to small sample sizes, focusing on issues like statistical power, reliability of p-values, and validity of assumptions. *BMC Medical Research Methodology*..1010  
* Maurer, D., et al. (2025). AutoEval-Phi: A Prediction-Powered Method for Automatic Evaluation of (Instruction-Following) Language Models. *arXiv preprint arXiv:2403.07008*. 80  
* McGill University. (n.d.). Prediction-powered inference: Evaluating efficiency in genomic data analysis..1818  
* Miao, W., Lei, L., Ramdas, A., & Zrnic, T. (2024). Active Inference: A New Methodology for Machine-Learning-Assisted Data Collection. *arXiv preprint arXiv:2403.03208*. 82  
* Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. *Statistics in Medicine, 38*(11), 2074-2102. 58 (This is a general guide for simulation studies).  
* Paxton, P., Curran, P. J., Bollen, K. A., Kirby, J., & Chen, F. (2001). Monte Carlo experiments: Design and implementation. *Structural Equation Modeling, 8*(2), 287-312. 57  
* ppi\_py Documentation. (n.d.). API Reference for PPI. 22  
* Quarto Documentation. (n.d.)..44  
* Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine Learning in Medicine. *New England Journal of Medicine, 380*(14), 1347-1358. 14 (Discusses ML vs. Statistics in medicine).  
* Schaefer, K. (2004). Too much of a good thing? The gravity model and the data-generating process. *USITC Office of Economics Working Paper*. 87  
* Semi-supervised learning general concepts. 3  
* Sim, J., & Wright, C. C. (2005). The Kappa Statistic in Reliability Studies: Use, Interpretation, and Sample Size Requirements. *Physical Therapy, 85*(3), 257-268..15  
* Timms, L., et al. (2024). How to check a simulation study. *International Journal of Epidemiology, 53*(1), dyad134. 59  
* Zhang, A., Brown, L. D., & Cai, T. T. (2018). Semi-supervised inference: General theory and estimation of means. *arXiv preprint arXiv:1712.00432*. 90 (Illustrates semi-supervised mean estimation and simulation setup).

## **附录 (可选)**

### **A.1. 更多数学推导**

(如果报告主体中省略了复杂的数学步骤，可在此处详细阐述，例如PPI++中λ参数的具体优化求解过程，或特定估计量方差的推导。)

### **A.2. 额外的模拟设置或结果**

(如果进行了额外的敏感性分析，例如改变标注集与未标注集的比例 nlab​/Nunlab​，或系统性地改变预训练模型 f(X) 的预测准确率，并将这些结果作为补充材料呈现。)

### **A.3. 扩展代码片段**

(如果报告主体中的代码片段为简洁起见有所省略，可在此处提供更完整的、用于说明关键实现细节的代码段。)

#### **Works cited**

1. Prediction-powered inference \- UBC Statistics, accessed on June 4, 2025, [https://www.stat.ubc.ca/\~john/papers/AngelopoulosScience2023.pdf](https://www.stat.ubc.ca/~john/papers/AngelopoulosScience2023.pdf)  
2. \[R\] Revisiting Semi-Supervised Learning in the Era of Foundation Models \- Reddit, accessed on June 4, 2025, [https://www.reddit.com/r/MachineLearning/comments/1jg0up9/r\_revisiting\_semisupervised\_learning\_in\_the\_era/](https://www.reddit.com/r/MachineLearning/comments/1jg0up9/r_revisiting_semisupervised_learning_in_the_era/)  
3. Semi-Supervised Learning in ML \- GeeksforGeeks, accessed on June 4, 2025, [https://www.geeksforgeeks.org/ml-semi-supervised-learning/](https://www.geeksforgeeks.org/ml-semi-supervised-learning/)  
4. Prediction-Powered E-Values \- arXiv, accessed on June 4, 2025, [https://arxiv.org/html/2502.04294v2](https://arxiv.org/html/2502.04294v2)  
5. Semi-Supervised Learning, Explained with Examples \- AltexSoft, accessed on June 4, 2025, [https://www.altexsoft.com/blog/semi-supervised-learning/](https://www.altexsoft.com/blog/semi-supervised-learning/)  
6. Semi-Supervised Learning: Techniques & Examples \[2024\] \- V7 Labs, accessed on June 4, 2025, [https://www.v7labs.com/blog/semi-supervised-learning-guide](https://www.v7labs.com/blog/semi-supervised-learning-guide)  
7. M-statistics Optimal Statistical Inference for a Small Sample, accessed on June 4, 2025, [https://wright.ecampus.com/mstatistics-optimal-statistical-inference/bk/9781119891796](https://wright.ecampus.com/mstatistics-optimal-statistical-inference/bk/9781119891796)  
8. Statistical inference \- Wikipedia, accessed on June 4, 2025, [https://en.wikipedia.org/wiki/Statistical\_inference](https://en.wikipedia.org/wiki/Statistical_inference)  
9. More about the basic assumptions of t-test: normality and sample size \- PMC, accessed on June 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6676026/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6676026/)  
10. A solution to minimum sample size for regressions \- PMC \- PubMed Central, accessed on June 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7034864/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7034864/)  
11. Regression with very small sample size \- Cross Validated \- Stack Exchange, accessed on June 4, 2025, [https://stats.stackexchange.com/questions/116132/regression-with-very-small-sample-size](https://stats.stackexchange.com/questions/116132/regression-with-very-small-sample-size)  
12. A solution to minimum sample size for regressions \- PMC, accessed on June 4, 2025, [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7034864/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7034864/)  
13. Statistical inference using machine learning and classical techniques based on accumulated local effects (ALE) \- arXiv, accessed on June 4, 2025, [https://arxiv.org/html/2310.09877v2](https://arxiv.org/html/2310.09877v2)  
14. Statistics versus machine learning \- PMC, accessed on June 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6082636/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6082636/)  
15. Machine Learning Predictions as Regression Covariates | Political Analysis, accessed on June 4, 2025, [https://www.cambridge.org/core/journals/political-analysis/article/machine-learning-predictions-as-regression-covariates/462A74A46A97C20A17CF640BDA72B826](https://www.cambridge.org/core/journals/political-analysis/article/machine-learning-predictions-as-regression-covariates/462A74A46A97C20A17CF640BDA72B826)  
16. 10.10 \- Other Regression Pitfalls | STAT 462, accessed on June 4, 2025, [https://online.stat.psu.edu/stat462/node/185/](https://online.stat.psu.edu/stat462/node/185/)  
17. Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of Large Language Models \- arXiv, accessed on June 4, 2025, [https://arxiv.org/html/2306.04746v3](https://arxiv.org/html/2306.04746v3)  
18. Prediction-Powered Inference: Evaluating ... \- McGill University, accessed on June 4, 2025, [https://www.mcgill.ca/ose/files/ose/prediction-powered\_inference\_evaluating\_efficiency\_in\_genomic\_data\_analysis.pdf](https://www.mcgill.ca/ose/files/ose/prediction-powered_inference_evaluating_efficiency_in_genomic_data_analysis.pdf)  
19. Prediction-powered Inference for Clinical Trials \- medRxiv, accessed on June 4, 2025, [https://www.medrxiv.org/content/10.1101/2025.01.15.25320578v1.full-text](https://www.medrxiv.org/content/10.1101/2025.01.15.25320578v1.full-text)  
20. arxiv.org, accessed on June 4, 2025, [https://arxiv.org/abs/2311.01453](https://arxiv.org/abs/2311.01453)  
21. PPI++: Efficient Prediction-Powered Inference \- arXiv, accessed on June 4, 2025, [https://arxiv.org/html/2311.01453v2](https://arxiv.org/html/2311.01453v2)  
22. API Reference for PPI — ppi\_py 0.2 documentation \- Read the Docs, accessed on June 4, 2025, [https://ppi-py.readthedocs.io/en/stable/ppi.html](https://ppi-py.readthedocs.io/en/stable/ppi.html)  
23. Prediction-powered inference | Request PDF \- ResearchGate, accessed on June 4, 2025, [https://www.researchgate.net/publication/375523918\_Prediction-powered\_inference](https://www.researchgate.net/publication/375523918_Prediction-powered_inference)  
24. Prediction-powered Inference for Clinical Trials | medRxiv, accessed on June 4, 2025, [https://www.medrxiv.org/content/10.1101/2025.01.15.25320578v1](https://www.medrxiv.org/content/10.1101/2025.01.15.25320578v1)  
25. The Alzheimer's Disease Neuroimaging Initiative: A review of papers published since its inception \- PubMed Central, accessed on June 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4108198/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4108198/)  
26. Structural MRI-based detection of Alzheimer's disease using feature ranking and classification error \- ADNI, accessed on June 4, 2025, [https://adni.loni.usc.edu/adni-publications/Structural%20MRI-based%20detection%20of%20Alzheimer's.pdf](https://adni.loni.usc.edu/adni-publications/Structural%20MRI-based%20detection%20of%20Alzheimer's.pdf)  
27. Overfitting, Model Tuning, and Evaluation of Prediction Performance \- NCBI, accessed on June 4, 2025, [https://www.ncbi.nlm.nih.gov/books/NBK583970/](https://www.ncbi.nlm.nih.gov/books/NBK583970/)  
28. Impact on bias mitigation algorithms to variations in inferred sensitive attribute uncertainty, accessed on June 4, 2025, [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1520330/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1520330/full)  
29. aangelopoulos/prediction-powered-inference: A statistical ... \- GitHub, accessed on June 4, 2025, [https://github.com/aangelopoulos/prediction-powered-inference](https://github.com/aangelopoulos/prediction-powered-inference)  
30. aangelopoulos/ppi\_py: A package for statistically rigorous ... \- GitHub, accessed on June 4, 2025, [https://github.com/aangelopoulos/ppi\_py](https://github.com/aangelopoulos/ppi_py)  
31. Prediction-Powered Inference with Imputed Covariates and Nonuniform Sampling \- arXiv, accessed on June 4, 2025, [https://arxiv.org/abs/2501.18577](https://arxiv.org/abs/2501.18577)  
32. \[Literature Review\] Prediction-Powered Inference with Imputed Covariates and Nonuniform Sampling \- Moonlight, accessed on June 4, 2025, [https://www.themoonlight.io/en/review/prediction-powered-inference-with-imputed-covariates-and-nonuniform-sampling](https://www.themoonlight.io/en/review/prediction-powered-inference-with-imputed-covariates-and-nonuniform-sampling)  
33. Cursor docs-Cursor Documentation-Cursor ai documentation \- Cursor中文文档, accessed on June 4, 2025, [https://cursordocs.com/en](https://cursordocs.com/en)  
34. Top Features of Cursor AI \- APPWRK, accessed on June 4, 2025, [https://appwrk.com/cursor-ai-features](https://appwrk.com/cursor-ai-features)  
35. Python \- Cursor, accessed on June 4, 2025, [https://docs.cursor.com/guides/languages/python](https://docs.cursor.com/guides/languages/python)  
36. Cursor AI: A Guide With 10 Practical Examples \- DataCamp, accessed on June 4, 2025, [https://www.datacamp.com/tutorial/cursor-ai-code-editor](https://www.datacamp.com/tutorial/cursor-ai-code-editor)  
37. How to Use Cursor Jupyter Notebook \- Apidog, accessed on June 4, 2025, [https://apidog.com/blog/cursor-jupyter-notebook/](https://apidog.com/blog/cursor-jupyter-notebook/)  
38. Cursor IDE for Jupyter Notebooks: Data Science and Analysis Tasks \- Reddit, accessed on June 4, 2025, [https://www.reddit.com/r/cursor/comments/1jdatdg/cursor\_ide\_for\_jupyter\_notebooks\_data\_science\_and/](https://www.reddit.com/r/cursor/comments/1jdatdg/cursor_ide_for_jupyter_notebooks_data_science_and/)  
39. yang3kc/cursor\_latex\_template: Cursor configuration for LaTeX projects \- GitHub, accessed on June 4, 2025, [https://github.com/yang3kc/cursor\_latex\_template](https://github.com/yang3kc/cursor_latex_template)  
40. Cursor for writing markdown documentation \- Reddit, accessed on June 4, 2025, [https://www.reddit.com/r/cursor/comments/1ip7vwd/cursor\_for\_writing\_markdown\_documentation/](https://www.reddit.com/r/cursor/comments/1ip7vwd/cursor_for_writing_markdown_documentation/)  
41. Top 10 Python Libraries for Data Analysis \- MarkTechPost, accessed on June 4, 2025, [https://www.marktechpost.com/2024/11/11/top-10-python-libraries-for-data-analysis/](https://www.marktechpost.com/2024/11/11/top-10-python-libraries-for-data-analysis/)  
42. Converting Pandas DataFrames to LaTeX and Markdown Tables \- Tilburg Science Hub, accessed on June 4, 2025, [https://tilburgsciencehub.com/topics/visualization/reporting-tables/reportingtables/pandas-latex-tables/](https://tilburgsciencehub.com/topics/visualization/reporting-tables/reportingtables/pandas-latex-tables/)  
43. Reproducible research and data analysis \- Open Science \- LibGuides at University of the Free State, accessed on June 4, 2025, [https://ufs.libguides.com/c.php?g=983440\&p=8841851](https://ufs.libguides.com/c.php?g=983440&p=8841851)  
44. Quarto Cheat Sheet (Previously Known as RMarkdown) \- DataCamp, accessed on June 4, 2025, [https://www.datacamp.com/cheat-sheet/quarto-cheat-sheet-previously-known-as-r-markdown](https://www.datacamp.com/cheat-sheet/quarto-cheat-sheet-previously-known-as-r-markdown)  
45. Guide – Quarto, accessed on June 4, 2025, [https://quarto.org/docs/guide/](https://quarto.org/docs/guide/)  
46. Reproducible Stats Reports with R Markdown \- Number Analytics, accessed on June 4, 2025, [https://www.numberanalytics.com/blog/reproducible-stats-reports-rmarkdown](https://www.numberanalytics.com/blog/reproducible-stats-reports-rmarkdown)  
47. Tables – Quarto, accessed on June 4, 2025, [https://quarto.org/docs/authoring/tables.html](https://quarto.org/docs/authoring/tables.html)  
48. Great Tables: Generating LaTeX Output for PDF \- GitHub Pages, accessed on June 4, 2025, [https://posit-dev.github.io/great-tables/blog/latex-output-tables/](https://posit-dev.github.io/great-tables/blog/latex-output-tables/)  
49. Discover great\_tables: The Python Answer to R's {gt} Package for Table Formatting in Quarto and Shiny for Python \- Appsilon, accessed on June 4, 2025, [https://www.appsilon.com/post/great-tables](https://www.appsilon.com/post/great-tables)  
50. Figures – Quarto, accessed on June 4, 2025, [https://quarto.org/docs/authoring/figures.html](https://quarto.org/docs/authoring/figures.html)  
51. Cross References \- Quarto, accessed on June 4, 2025, [https://quarto.org/docs/authoring/cross-references.html](https://quarto.org/docs/authoring/cross-references.html)  
52. Book Crossrefs \- Quarto, accessed on June 4, 2025, [https://quarto.org/docs/books/book-crossrefs.html](https://quarto.org/docs/books/book-crossrefs.html)  
53. Appendix B — Introduction to Quarto and Markdown – GOG422/522: GIS For Social Sciences, accessed on June 4, 2025, [https://www.albany.edu/spatial/GIS4SS/lecture/91-quarto-more.html](https://www.albany.edu/spatial/GIS4SS/lecture/91-quarto-more.html)  
54. Citations \- Quarto, accessed on June 4, 2025, [https://quarto.org/docs/authoring/citations.html](https://quarto.org/docs/authoring/citations.html)  
55. Quarto Reference Manager: Simple citation management with CiteDrive for RStudio, accessed on June 4, 2025, [https://www.citedrive.com/en/quarto/](https://www.citedrive.com/en/quarto/)  
56. What is Reproducible Research? \- Displayr, accessed on June 4, 2025, [https://www.displayr.com/what-is-reproducible-research/](https://www.displayr.com/what-is-reproducible-research/)  
57. Monte Carlo Experiments: Design and Implementation \- Patrick J. Curran, Ph.D., accessed on June 4, 2025, [https://curran.web.unc.edu/wp-content/uploads/sites/6785/2015/03/PaxtonCurranBollenKirbyChen2001.pdf](https://curran.web.unc.edu/wp-content/uploads/sites/6785/2015/03/PaxtonCurranBollenKirbyChen2001.pdf)  
58. Using simulation studies to evaluate statistical methods \- PMC, accessed on June 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/)  
59. How to check a simulation study | International Journal of Epidemiology \- Oxford Academic, accessed on June 4, 2025, [https://academic.oup.com/ije/article/53/1/dyad134/7313663](https://academic.oup.com/ije/article/53/1/dyad134/7313663)  
60. Guidance for the Conduct and Reporting of Modeling and Simulation Studies in the Context of Health Technology Assessment, accessed on June 4, 2025, [https://effectivehealthcare.ahrq.gov/products/decision-models-guidance/methods](https://effectivehealthcare.ahrq.gov/products/decision-models-guidance/methods)  
61. How to check a simulation study \- ResearchGate, accessed on June 4, 2025, [https://www.researchgate.net/publication/375386382\_How\_to\_check\_a\_simulation\_study](https://www.researchgate.net/publication/375386382_How_to_check_a_simulation_study)  
62. KIPA2024\_01 Simulation Analysis Report \- Organ Procurement and Transplantation Network, accessed on June 4, 2025, [https://optn.transplant.hrsa.gov/media/baldrgot/ki2024\_01\_request\_analysis\_report.pdf](https://optn.transplant.hrsa.gov/media/baldrgot/ki2024_01_request_analysis_report.pdf)  
63. Mean squared error \- Wikipedia, accessed on June 4, 2025, [https://en.wikipedia.org/wiki/Mean\_squared\_error](https://en.wikipedia.org/wiki/Mean_squared_error)  
64. 7 Crucial Stats: How Mean Squared Error Impacts Model Precision \- Number Analytics, accessed on June 4, 2025, [https://www.numberanalytics.com/blog/mse-models-7-stats](https://www.numberanalytics.com/blog/mse-models-7-stats)  
65. What Is a Confidence Interval and How Do You Calculate It? \- Investopedia, accessed on June 4, 2025, [https://www.investopedia.com/terms/c/confidenceinterval.asp](https://www.investopedia.com/terms/c/confidenceinterval.asp)  
66. Confidence interval \- Wikipedia, accessed on June 4, 2025, [https://en.wikipedia.org/wiki/Confidence\_interval](https://en.wikipedia.org/wiki/Confidence_interval)  
67. The Confusion Matrix in Hypothesis Testing \- Towards Data Science, accessed on June 4, 2025, [https://towardsdatascience.com/the-confusion-matrix-explained-part-1-5513c6f659c1/](https://towardsdatascience.com/the-confusion-matrix-explained-part-1-5513c6f659c1/)  
68. Hypothesis Testing | GeeksforGeeks, accessed on June 4, 2025, [https://www.geeksforgeeks.org/understanding-hypothesis-testing/](https://www.geeksforgeeks.org/understanding-hypothesis-testing/)  
69. Prediction-Powered Inference for Clinical Trials \- PDXScholar, accessed on June 4, 2025, [https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1429\&context=mth\_fac](https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1429&context=mth_fac)  
70. Cross-prediction-powered inference | PNAS, accessed on June 4, 2025, [https://www.pnas.org/doi/10.1073/pnas.2322083121](https://www.pnas.org/doi/10.1073/pnas.2322083121)  
71. Cross-prediction-powered inference \- ResearchGate, accessed on June 4, 2025, [https://www.researchgate.net/publication/379540582\_Cross-prediction-powered\_inference](https://www.researchgate.net/publication/379540582_Cross-prediction-powered_inference)  
72. proceedings.neurips.cc, accessed on June 4, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/c9fcd02e6445c7dfbad6986abee53d0d-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/c9fcd02e6445c7dfbad6986abee53d0d-Paper-Conference.pdf)  
73. arxiv.org, accessed on June 4, 2025, [https://arxiv.org/abs/2502.04294](https://arxiv.org/abs/2502.04294)  
74. Prediction-Powered E-Values \- arXiv, accessed on June 4, 2025, [https://arxiv.org/pdf/2502.04294](https://arxiv.org/pdf/2502.04294)  
75. Prediction-powered inference \- Clara Wong-Fannjiang, accessed on June 4, 2025, [https://clarafy.github.io/data/ABFJZ2023Science.pdf](https://clarafy.github.io/data/ABFJZ2023Science.pdf)  
76. Cross-Prediction-Powered Inference \- arXiv, accessed on June 4, 2025, [https://arxiv.org/html/2309.16598v3](https://arxiv.org/html/2309.16598v3)  
77. Stratified Prediction-Powered Inference for Effective Hybrid Evaluation of Language Models, accessed on June 4, 2025, [https://neurips.cc/virtual/2024/poster/96386](https://neurips.cc/virtual/2024/poster/96386)  
78. \[2306.04746\] Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of Large Language Models \- arXiv, accessed on June 4, 2025, [https://arxiv.org/abs/2306.04746](https://arxiv.org/abs/2306.04746)  
79. Nathan Kallus, accessed on June 4, 2025, [https://nathankallus.com/](https://nathankallus.com/)  
80. AutoEval Done Right: Using Synthetic Data for Model Evaluation \- arXiv, accessed on June 4, 2025, [https://arxiv.org/html/2403.07008v1](https://arxiv.org/html/2403.07008v1)  
81. AutoEval Done Right: Using Synthetic Data for Model Evaluation \- arXiv, accessed on June 4, 2025, [https://arxiv.org/html/2403.07008v2](https://arxiv.org/html/2403.07008v2)  
82. Active Statistical Inference \- arXiv, accessed on June 4, 2025, [https://arxiv.org/html/2403.03208v1](https://arxiv.org/html/2403.03208v1)  
83. arXiv:2403.03208v2 \[stat.ML\] 29 May 2024, accessed on June 4, 2025, [https://arxiv.org/pdf/2403.03208](https://arxiv.org/pdf/2403.03208)  
84. References and cross-references \- Jupyter Book, accessed on June 4, 2025, [https://jupyterbook.org/content/references.html](https://jupyterbook.org/content/references.html)  
85. Dashboard Data Display \- Quarto, accessed on June 4, 2025, [https://quarto.org/docs/dashboards/data-display.html](https://quarto.org/docs/dashboards/data-display.html)  
86. accessed on January 1, 1970, [https://quarto.org/docs/output-formats/pdf-customization.html](https://quarto.org/docs/output-formats/pdf-customization.html)  
87. Monte Carlos Appraisals of Gravity Model Specifications \- USITC, accessed on June 4, 2025, [https://www.usitc.gov/publications/332/ec200405a.pdf](https://www.usitc.gov/publications/332/ec200405a.pdf)  
88. Semi-Supervised Learning: Techniques & Examples \- StrataScratch, accessed on June 4, 2025, [https://www.stratascratch.com/blog/semi-supervised-learning-techniques-and-examples/](https://www.stratascratch.com/blog/semi-supervised-learning-techniques-and-examples/)  
89. Lab: Build a Semi-Supervised Learning Model \- DataRobot University, accessed on June 4, 2025, [https://learn.datarobot.com/courses/lab-build-a-semi-supervised-learning-model](https://learn.datarobot.com/courses/lab-build-a-semi-supervised-learning-model)  
90. www-stat.wharton.upenn.edu, accessed on June 4, 2025, [http://www-stat.wharton.upenn.edu/\~tcai/paper/Semisupervised-Means.pdf](http://www-stat.wharton.upenn.edu/~tcai/paper/Semisupervised-Means.pdf)  
91. Template for data science projects \- Kaggle, accessed on June 4, 2025, [https://www.kaggle.com/code/precisionmed/template-for-data-science-projects](https://www.kaggle.com/code/precisionmed/template-for-data-science-projects)