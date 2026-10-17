# CAMUS 心脏超声多结构分割（2026-10-12）

## 任务与来源
按用户提供的下一阶段要求开展新训练：灰度心脏超声、左心室腔/心肌/左心房三结构分割；保留低质量样本，生成轮廓、ED/ES对照、面积与双视图体积/EF参考结果。DeepSeek执行，Codex规划与核验；旧皮肤模型交付完整保留。

数据来自 [CAMUS官方](https://www.creatis.insa-lyon.fr/Challenge/camus/) 与 [Human Heart Project公共库](https://humanheart-project.creatis.insa-lyon.fr/database/collection/6373703d73e9f0047faa1bc8)。当前公开NIfTI版500患者，官方精确划分400训练/50验证/50测试，两个视角与ED/ES同患者不跨组。来源、许可、名单原件在runs/role_a_camus/20261012/preparation/source_review。许可CC BY-NC-SA4.0，非商业科研，引用Leclerc等IEEE TMI2019，DOI10.1109/TMI.2019.2900516。

## 已核验的数据与工程
- ZIP 3835077244字节，SHA862474ff61a660d0d0e9fe9867d5189b8536406e266d983a9914a9cb165c0c35；7508成员/500患者，CRC及安全路径全部通过。
- 完整训练缓存1600帧、验证缓存200帧，总499820343字节；项目当前约20.77GB，40GB以内。combined SHA3112df3f46856ec36a5b42a6fa1752063d8851ef982c07c9b9fec5769cb4b217。
- Codex逐帧独立核验1800个缓存SHA、全部原尺寸图像/GT逐像素与官方原件一致、spacing/affine/人工quality与患者隔离。数据准备仅解码训练/验证GT；模型冻结后依用户明确授权，一次解码官方测试50患者用于最终验收，完成后放行撤销。
- Good/Medium/Poor训练帧816/604/180；验证帧100/78/22。没有按亮度伪造质量分组，没有剔除Poor。
- 真实30项合成检查与实际读取集成核验通过；31st检查覆盖AMP大批次FP32求和。旧失败记录CPU01/02、准备01–03保留；不能据它们宣称完成。
- 体积计算提取官方20disk/AV-base/apex/isotropic双视图算法，非对称/非等距mask与原算法逐值一致。四LV掩膜、每视图各自物理间距；空/破碎/几何异常返回null及原因，不以单视图平均或row-span替代官方结果。
- 单原生帧流式预处理；训练只按需读256输入/标签/valid。原生GT完整保留。模型Conv/ReLU/MaxPool/nearest上采样+Conv/concat/4类head，不用sigmoid。

## 正式训练与验证（已完成）
2026-10-12 08:39:58 UTC（北京时间16:39:58）由DeepSeek唯一启动首次baseline01，CUDA RTX4050，模型从头初始化。256单通道，4类输出，base16/3pool；AdamW学习率0.0003，batch8，AMP卷积+FP32 CE/Dice，最多100epoch、10800秒、patience15。原皮肤权重未用于初始化。初始选择并非永久限制训练方法。

选模主指标固定为原尺寸三前景Dice、帧等权后患者等权，只用官方验证50。最后完整epoch原子last/best断点含模型/优化器/调度器/AMP/RNG/源码/数据/config/epoch/best/stale；中途STOP保留完整断点、另存不可直接恢复的partial取证。超时/停止不冒充成功。

实际状态与结果入口：
- runs/role_a_camus/20261012/execution/train01/START.json、STATUS.json、RESULT.json、stdout.log、stderr.log
- runs/role_a_camus/20261012/baselines/baseline01/STATUS.json、last.pt、best.pt、BEST_DEV_RECORDS.json
- runs/role_a_camus/20261012/CODEX_CURRENT_PHASE_CHECKPOINT.json

## 已完成的交付验收
训练100epoch实际完成，child与wrapper均exit0；最佳第94轮，冻结best SHA dcfe16a59cdde4821180561aa2d8c72393f1f53e6c8bbbcdade0541413a1e616。模型从头训练于2026-10-12，未使用旧皮肤模型权重。

完整官方验证50患者/200帧评估exit0，native三前景患者等权Dice0.8798801934838069，LV0.9218524585、MYO0.8401830130、LA0.8776051089。官方Poor22帧Dice0.8397815083。HD95 LV7.9299mm/MYO7.9806mm/LA10.9793mm。全部原GT50患者可计算官方双视图体积/EF；原始预测仅24/50可算，26例LV碎裂多连通域给null与原因，没有删除失败或以0冒充EF。

交付在deliverables/role_a_camus_20261012与同名zip：纯FP32 model_state、完整报告/逐帧记录/50例几何、100张EDES图、代表/最差/Poor案例、3组8bitPNG/RAW/valid/letterbox/spacing/affine、训练曲线、引用许可和SHA清单。362个交付文件已逐一验SHA，纯权重可weights_only安全加载，30个tensor逐值等同冻结best；3份RAW均65536字节，与PNG逐像素一致。ZIP已验安全路径/CRC。FP32权重SHA74d0da6f13b48528b221e6c295d90de73b6e16f1320a860bc086a05e3c2b774f；最终ZIP SHAea696369b889cc14cab6435a594a9189878faf75af9624418298b8e3ab6ff0f0，39209164字节。

真实预算487060参数、2656043008 MAC/frame，FP32参数1948240B；理论INT8权重486352B+INT32bias2832B（未量化），最大单激活张量FP32约12MiB（合成零输入核实实际decoder concat192/96/48通道）；CUDA batch8完整验证峰值allocated310277632B/reserved490733568B。这些指标不代表PYNQ吞吐或板端验证。

## 一次独立测试与最终收尾（已完成）
用户明确批准一次最终测试，模型/方法已经冻结。test_eval01真实exit0，50患者/200帧，患者等权三前景Dice0.8827694442；LV0.9213098710、心肌0.8418114232、左心房0.8851870384。HD95分别7.8661/8.8790/8.4377mm；Poor30帧Dice0.8534774955，全病例保留。原GT50/50可算EF，原预测25/50可算，失败25例LV碎裂（19例2连通块、5例3个、1例4个），全部返回null和原因，未据测试重训或修饰预测。

Codex使用已保存NPZ独立复算600个结构Dice，与全部RECORDS/SUMMARY一致；PNG为0/85/170/255映射，NPZ真实标签0/1/2/3。实际测试放行已移至OWNER_TEST_RELEASE_consumed.json，ROOT文件不存在。早前自动审批拒绝是在尚无具体授权时发生，收到人类肯定授权后已合法完成；不存在待授权 blocker。最终362文件/363ZIP成员逐字节和SHA/CRC/路径复核exit0，模型/训练与评估源码不变。资源表错误由owner按实际30个张量和合成前向hook纠正，证据CODEX_RESOURCE_AUDIT；没有改动训练权重或任何测试数值。

最终交付报告同时包含独立测试记录、50例体积、100张EDES图与3组输入样例；验证版与修资源前版ZIP均保留。资源快照21063521340B<40GB，后续少量文书未计入。一次执行许可已关闭，CAMUS STOP在位，精确授权原件保存在closure。INT8权重、PYNQ板端及可选2.5D尚未完成；下一疾病/下一训练回合由用户另行明确。
