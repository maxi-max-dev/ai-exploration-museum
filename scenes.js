import {network,attention,toyLanguageStep,convolution,qTrain,mcts,diffuse} from './math.js?v=4';
const fmt=n=>Number(n).toFixed(3);
const text=(x,y,t,cls='')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="middle">${t}</text>`;
const edge=(x,y,a,b,on=false)=>`<path d="M${x} ${y} L${a} ${b}" class="scene-edge ${on?'travelling':''}"/>`;
const box=(x,y,w,h,t,on=false)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" class="scene-box ${on?'lit':''}"/>${text(x+w/2,y+h/2+6,t)}`;
const circle=(x,y,r,t,on=false)=>`<circle cx="${x}" cy="${y}" r="${r}" class="scene-box ${on?'lit':''}"/>${text(x,y+6,t)}`;
const wrap=body=>`<svg viewBox="0 0 640 310" role="img" aria-label="当前步骤的机制图，文字说明见下方">${body}</svg>`;
const sentence=['我','喜欢','学习','AI'];
const searchBoard=['X',null,null,null,'O',null,null,null,null];
const grid=qTrain(100,.2,-.04).last;
const diffusion=diffuse(1,6);
export const sceneFor=id=>({C03:'search',C07:'network',C08:'network',C10:'vision',C11:'language',C12:'reward',C13:'search',C14:'language',C15:'language',C16:'language',C17:'language',C18:'language',C19:'language',C21:'diffusion',C22:'retrieval'}[id]||null);
export const sceneDefinitions={
 search:{title:'看一次搜索怎样选出下一步',steps:['当前局面','展开候选','少量试走','增加预算','回传比较','选择落点'],note:'井字棋UCT真实计算，X先行。用于解释搜索循环；不是围棋或AlphaGo的神经网络、先验与搜索实现。',explain:['X已在左上，O在中央。现在轮到X；搜索从这个局面出发。','先列出空位。每条分支代表一个合法落子，并不保证每条都会赢。','运行8次搜索迭代，模拟到终局，把胜/和/负记为1/0.5/0。','把预算增到80次。反复选择、扩展、模拟与回传，各分支的访问数发生变化。','每个N是实际访问数，Q是从根玩家X视角的平均结果。对手回合选择会反转价值偏好。','在访问最多的分支落子。更多搜索能改善判断，但有限预算并不保证最优。']},
 network:{title:'让一个数字走过神经网络',steps:['输入','加权与偏置','非线性','输出与误差','反向梯度','更新参数'],note:'单隐藏单元的真实tanh计算，x=.6、w=.7、b=.2，输出权重=.8，目标=1。只更新w。',explain:['输入x=.6。连线上的权重决定它传过去时被放大或缩小多少。','先计算z=w×x+b=.7×.6+.2=.62。此时仍是线性变换。','tanh把z变成隐藏表示h。加入非线性，才能摆脱简单线性组合的限制。','输出y=.8h，与目标1比较。平方损失把当前误差变成一个标量。','沿原来的计算图反向使用链式法则，求出损失对w的导数。反向传播计算梯度，不直接替你选步长。','优化器执行w←w−.2×梯度，再重新前向。比较更新前后的损失；下一步训练会继续这个循环。']},
 language:{title:'看一句话怎样多出一个token',steps:['文字变向量','查询与匹配','汇入上下文','计算候选概率','采样一个token','接回输入'],note:'教学分词、二维手工参数与简化层；计算真实，但未训练，也不是商业模型的语言能力。',explain:['把“我 / 喜欢 / 学习 / AI”分成4个教学token，再转成数值向量，并加位置信息。','最后位置发出查询，与每个可见位置的键计算相似度。连线粗细表示真实计算的注意力权重。','按权重混合值向量，得到来自上下文的输出；注意力不是查找一个唯一答案。','经过残差、归一化与教学MLP，再投影到4个候选token。柱长表示softmax后的概率。','固定种子42，从概率分布抽取一个token。概率最高和每次必选不是同一件事。','把选中的token接回输入，再算一次。新的查询位置与权重改变，逐token生成就是这样的回路。']},
 vision:{title:'让卷积核走过一张小图',steps:['原图','左上窗口','向右滑动','完成首行','继续向下','输出特征图'],note:'5×5二值图、3×3竖边缘核，步长1、不填充。图中数值来自实验E05同一计算。',explain:['右侧三列像素为1，左侧两列为0。亮暗边界在哪里？','把3×3核放在左上窗口，对9对数字逐项相乘再相加。','窗口向右移动一格；同一组核权重在每个位置共享。','首行得到3、3、0：跨过明暗变化的位置响应较大，均匀区域为0。','继续逐行扫描，每个输出格对应输入的一块局部窗口。','得到3×3特征图。这里是手工核；CNN训练通常会从数据学习核参数。']},
 reward:{title:'跟着一次行动看奖励怎样回来',steps:['起点','第一步','第二步','第三步','第四步','这一轮结束'],note:'4×4 Q-learning训练100轮后的最后一条轨迹。绿色终点+1，赭色陷阱−1，每步−.04；固定种子。',explain:['智能体从左上角出发。行动之前并不知道这轮最后的结果。','选择一个动作，环境返回新位置与即时奖励；这不是监督学习里的标准动作标签。','使用奖励与下一状态的最大Q值更新原状态的动作价值。','价值学习把后续回报的信息逐渐传回较早的选择。','同一个环境里，探索概率会改变尝试过的路线；不探索可能错过更好的路。','跳到这轮轨迹的最后一步，最终到达右下终点。这里展示的是训练后的一条路径；改奖励或探索率可在E07重跑。']},
 diffusion:{title:'观察噪声怎样逐步被去掉',steps:['加入噪声','去噪一步','去噪两步','去噪三步','去噪四步','去噪六步'],note:'已知目标的插值去噪演示，不是训练好的扩散网络，也不是真实生成采样器。',explain:['原始目标是中央方块。添加固定种子的噪声，像素偏离目标。','每步把当前值向已知目标移动四分之一；误差因此缩小。','继续执行相同更新，中央形状逐渐可见。','比较两幅图：这里能直接访问干净目标，所以不能把效果当作真实生成能力。','真实扩散模型学习噪声/分数等预测，再配合采样器；未知目标不会直接提供给它。','玩具演示只帮助看懂逐步去噪的方向。完整机制、训练与采样的区别见本章。']},
 retrieval:{title:'把“查资料”这条路径拆开看',steps:['提出问题','检索片段','读取证据','组合上下文','生成回答','核对来源'],note:'固定语料的流程动画，未调用模型。动态冲突、无结果与工具失败可在E12操作。',explain:['问题是“本馆周一开门吗？”首先要查外部资料，不能只凭生成器猜。','从固定语料检索到开放时间片段。检索发生在回答时，没有修改模型参数。','读取片段：“周二至周日开放，周一闭馆。”这才是当前回答的证据。','把问题和相关片段组合进上下文。无关文档即使检索到了，也不能支撑结论。','演示回答：“周一闭馆。”这里是固定模板；真实系统还要判断冲突和资料时效。','把结论链接回片段。能给引用不等于引用真的支持结论，还需要核对内容。']}
};
export function sceneFrame(kind,step){
 const active=step;
 if(kind==='network'){
  const n=network(.6,.7),w=.7-.2*n.grad,next=network(.6,w),vals=['.600',fmt(step===5?next.z:n.z),fmt(step===5?next.h:n.h),fmt(step===5?next.y:n.y)],xs=[65,235,405,575];
  let body=xs.slice(0,3).map((x,i)=>edge(x+31,105,xs[i+1]-31,105,step===i+1||step===4).replace('class=','style="animation-direction:'+ (step===4?'reverse':'normal') +'" class=')).join('');
  body+=xs.map((x,i)=>circle(x,105,31,vals[i],step===i||step===4||step===5)+text(x,163,['输入 x','加权和 z','隐藏层 h','输出 y'][i])).join('');
  body+=text(320,36,step===4?'梯度沿计算图反向传播':'数值沿计算图向前传播','scene-large');
  body+=box(65,205,220,52,step===5?`新 w = ${fmt(w)}`:`w = .700`,step>=4)+box(345,205,230,52,`损失 = ${fmt(step===5?next.loss:n.loss)}`,step>=3);
  if(step===4)body+=text(320,290,`∂L/∂w = ${fmt(n.grad)}；下一步使用学习率 .2`);
  return wrap(body);
 }
 if(kind==='search'){
  const budget=step<2?0:step===2?8:80,r=mcts(searchBoard,budget),chosen=r.choice,shown=[1,2,chosen??3],children=shown.map(i=>r.root.children.find(c=>c.move===i));
  let b='';for(let i=0;i<9;i++){let x=20+i%3*47,y=70+Math.floor(i/3)*47;b+=`<rect x="${x}" y="${y}" width="44" height="44" class="board-cell"/>`;if(!searchBoard[i]&&!(step===5&&i===chosen))b+=text(x+22,y+29,i+1,'square-number');if(searchBoard[i]||step===5&&i===chosen)b+=text(x+22,y+30,searchBoard[i]||'X',step===5&&i===chosen?'new-stone':'');}
  b+=text(86,245,step===5?'实际选中落点':'根局面：X行动')+circle(417,55,31,`N=${budget}`,step===0||step===4);
  const legal=shown;
  if(step>0){b+=[285,417,550].map((x,i)=>edge(417,86,x,153,step===2||step===4)+circle(x,177,28,`${legal[i]+1}号`,step===5&&chosen===legal[i])+text(x,231,children[i]?`N=${children[i].n}`:'N=0')+text(x,259,children[i]?.n?`Q=${fmt(children[i].w/children[i].n)}`:'Q=—')).join('');b+=text(417,296,'仅展示3条分支；计算包含全部7个合法落点');}
  return wrap(b);
 }
 if(kind==='language'){
  const initial=toyLanguageStep(sentence),tokens=step===5?[...sentence,initial.token]:sentence,r=step===5?toyLanguageStep(tokens,43):initial,ar=attention(tokens,tokens.length-1,true);let b='';
  tokens.forEach((t,i)=>{let x=30+i*116;b+=box(x,20,104,50,t,step===0||step===5&&i===4);if(step>=1&&step<3)b+=`<path d="M${x+52} 70L320 142" class="scene-edge travelling" style="stroke-width:${2+ar.weights[i]*12}"/>`});
  if(step<3){b+=box(202,143,236,55,step===0?'向量 + 位置信息':step===1?'最后位置发出查询':'加权混合 V',true);b+=text(320,242,step===2?`注意力输出 [${r.output.map(fmt).join(', ')}]`:'每个位置都保持自己的表示');}
  else{r.vocab.forEach((v,i)=>{let x=70+i*145,h=r.probs[i]*210;b+=`<rect x="${x}" y="${252-h}" width="60" height="${h}" class="prob-bar ${step>=4&&i===r.chosen?'selected':''}"/>`+text(x+30,277,v)+text(x+30,238-h,`${(r.probs[i]*100).toFixed(1)}%`);});b+=text(320,112,step===3?'候选token概率':step===4?`采样选中「${r.token}」`:'接回序列，下一轮重新计算');}
  return wrap(b);
 }
 if(kind==='vision'){
  const pixels=Array.from({length:25},(_,i)=>i%5>=2?1:0),conv=convolution(pixels,[-1,0,1,-1,0,1,-1,0,1]),index=[-1,0,1,2,5,8][step];let b='';
  pixels.forEach((v,i)=>{b+=`<rect x="${34+i%5*40}" y="${40+Math.floor(i/5)*40}" width="38" height="38" fill="${v?'#bfd7ca':'#314d47'}"/>`;});
  if(index>=0)b+=`<rect x="${34+index%3*40}" y="${40+Math.floor(index/3)*40}" width="118" height="118" class="scan-window"/>`;
  b+=edge(260,143,365,143,step>0);for(let i=0;i<9;i++)b+=box(390+i%3*56,62+Math.floor(i/3)*56,52,52,i<=index?conv[i].value:'·',i===index);
  b+=text(134,280,'输入 5 × 5')+text(472,280,'特征图 3 × 3');return wrap(b);
 }
 if(kind==='reward'){
  const idx=step===5?grid.length:Math.min(step,grid.length),pos=idx?grid[idx-1].ns:0;let b='';for(let i=0;i<16;i++)b+=`<rect x="${55+i%4*53}" y="${30+Math.floor(i/4)*53}" width="49" height="49" rx="4" fill="${i===15?'#bdddc5':i===5?'#ecc7ac':'#e6ebe3'}"/>`;
  b+=circle(79+pos%4*53,54+Math.floor(pos/4)*53,17,'●',true)+text(134,278,'走格子：从左上到右下');
  b+=text(445,86,`已执行 ${idx} 个动作`,'scene-large')+text(445,139,`即时奖励 ${idx?grid[idx-1].reward:'—'}`)+text(445,186,`累计奖励 ${fmt(grid.slice(0,idx).reduce((s,t)=>s+t.reward,0))}`)+text(445,237,idx===grid.length?'本轮轨迹完成':'探索、行动、观察、更新');return wrap(b);
 }
 if(kind==='diffusion'){
  const frame=diffusion.frames[[0,1,2,3,4,6][step]],err=frame.reduce((s,v,i)=>s+(v-diffusion.target[i])**2,0)/64;let b='';
  [frame,diffusion.target].forEach((a,k)=>a.forEach((v,i)=>{const c=Math.round(Math.max(0,Math.min(1,v))*220+20);b+=`<rect x="${60+k*330+i%8*24}" y="${36+Math.floor(i/8)*24}" width="24" height="24" fill="rgb(${c},${c},${c})"/>`}));
  b+=text(156,265,`当前误差 ${fmt(err)}`)+text(486,265,'已知干净目标');return wrap(b);
 }
 let b='';const labels=['问题','检索','片段','上下文','回答','来源'],positions=[[24,25],[234,25],[444,25],[444,180],[234,180],[24,180]];for(let i=0;i<6;i++){let [x,y]=positions[i];if(i<5){let [a,z]=positions[i+1];b+=edge(x+85,y+37,a+85,z+37,step===i+1);}b+=box(x,y,170,75,labels[i],i===active);}b+=text(320,145,step<2?'周一开门吗？':step<4?'证据：周二至周日开放，周一闭馆。':'回答：周一闭馆。引用：开放时间片段。');return wrap(b);
}
export function mountScene(el,id){
 const kind=sceneFor(id);if(!kind)return ()=>{};const config=sceneDefinitions[kind];let step=0,timer=null,stopped=false;const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 el.className='mechanism-player';el.innerHTML=`<div class="scene-heading"><h2>${config.title}</h2><span>动态导览 · 6步</span></div><div class="scene-canvas"></div><div class="scene-caption" aria-live="polite"></div><div class="scene-controls"><button data-play>播放过程</button><button class="secondary" data-prev>上一步</button><button class="secondary" data-next>下一步</button><button class="secondary" data-reset>重播</button><label>速度 <select data-speed><option value="2200">慢速</option><option value="1400" selected>正常</option><option value="800">快速</option></select></label><label><input type="checkbox" data-still ${reduced.matches?'checked':''}> 减弱动效</label></div><div class="scene-timeline" aria-label="选择演示步骤">${config.steps.map((s,i)=>`<button class="secondary" data-frame="${i}">${i+1}. ${s}</button>`).join('')}</div><p class="source-note">${config.note}</p>`;
 const play=el.querySelector('[data-play]'),still=el.querySelector('[data-still]');
 function draw(){el.classList.toggle('still',still.checked);el.querySelector('.scene-canvas').innerHTML=sceneFrame(kind,step);el.querySelector('.scene-caption').innerHTML=`<strong>${step+1} / 6 · ${config.steps[step]}</strong><p>${config.explain[step]}</p>`;el.querySelector('[data-prev]').disabled=step===0;el.querySelector('[data-next]').disabled=step===5;el.querySelectorAll('[data-frame]').forEach(b=>b.setAttribute('aria-current',+b.dataset.frame===step?'step':'false'));}
 function pause(){clearTimeout(timer);timer=null;play.textContent=step===5?'再看一遍':'播放过程';play.setAttribute('aria-pressed','false');el.classList.remove('playing');}
 function tick(){timer=setTimeout(()=>{if(stopped||!el.isConnected){pause();return;}step++;draw();if(step===5)pause();else tick();},+el.querySelector('[data-speed]').value);}
 function start(){if(step===5)step=0;draw();play.textContent='暂停';play.setAttribute('aria-pressed','true');el.classList.add('playing');tick();}
 play.onclick=()=>timer?pause():start();el.querySelector('[data-prev]').onclick=()=>{pause();step=Math.max(0,step-1);draw();};el.querySelector('[data-next]').onclick=()=>{pause();step=Math.min(5,step+1);draw();};el.querySelector('[data-reset]').onclick=()=>{pause();step=0;draw();start();};el.querySelectorAll('[data-frame]').forEach(b=>b.onclick=()=>{pause();step=+b.dataset.frame;draw();});still.onchange=()=>{pause();draw();};el.querySelector('[data-speed]').onchange=()=>{if(timer){pause();start();}};
 const onHidden=()=>{if(document.hidden)pause();},onReduce=()=>{still.checked=reduced.matches;pause();draw();};document.addEventListener('visibilitychange',onHidden);reduced.addEventListener('change',onReduce);const observer=new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)pause();},{threshold:.15});observer.observe(el);draw();
 return ()=>{stopped=true;pause();observer.disconnect();document.removeEventListener('visibilitychange',onHidden);reduced.removeEventListener('change',onReduce);};
}
