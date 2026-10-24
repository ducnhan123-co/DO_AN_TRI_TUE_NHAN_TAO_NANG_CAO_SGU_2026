/* Connect 4 RL Lab — game port + source explainer */

/* ------------------------------------------------------------------ */
/* Files & annotations                                                 */
/* ------------------------------------------------------------------ */

const FILES = [
  { name: 'README.md',   ext: 'md', meta: 'mục tiêu, luật, cách chạy' },
  { name: 'connect4.py', ext: 'py', meta: 'Gym env + reward + pygame' },
  { name: 'heuristic.py',ext: 'py', meta: 'thắng → chặn → UCB' },
  { name: 'minimax.py',  ext: 'py', meta: 'alpha-beta, depth 4' },
  { name: 'ensemble.py', ext: 'py', meta: 'bỏ phiếu đa số' },
  { name: 'play.py',     ext: 'py', meta: 'chơi bằng pygame' },
  { name: 'train.py',    ext: 'py', meta: 'PPO curriculum' },
  { name: 'test.py',     ext: 'py', meta: 'đo winrate' },
];

const SECTIONS = {
  'README.md': [
    { id:'r-intro', start:1, end:16, title:'Mục tiêu & luật chơi',
      body:'<p>Dự án dùng Reinforcement Learning để chơi Connect 4: bàn <strong>6×7</strong>, hai người thay phiên thả quân vào cột, ai nối đủ 4 quân trước thì thắng.</p><p>Mục tiêu của đồ án: cài đặt, huấn luyện và so sánh nhiều cách tiếp cận RL.</p>',
      tags:['6×7','2 người','4 liên tiếp'] },
    { id:'r-state', start:18, end:32, title:'State & reward trong README',
      body:'<p>Bàn cờ mã hóa <code>0/1/2</code>, 7 action ứng với 7 cột. README ghi thắng <strong>+10</strong>, chặn <strong>+3</strong>, nước sai <strong>−50</strong>, hòa <strong>0</strong>.</p><p><span class="warn">Lệch với code:</span> <code>connect4.py</code> đang đặt <code>DRAW_REWARD = 1</code> và <code>ILLEGAL_MOVE_PENALTY = −100</code>. Khi làm báo cáo, nên lấy code làm chuẩn và ghi chú chỗ lệch.</p>',
      tags:['reward shaping','README vs code'] },
    { id:'r-env', start:34, end:40, title:'Môi trường & render',
      body:'<p>Env là custom Gymnasium: quan sát 6×7, action rời rạc 7 cột, render bằng Pygame. Đối thủ mặc định là random, có thể gắn heuristic hoặc model đã train.</p>',
      tags:['Gymnasium','Pygame'] },
    { id:'r-agents', start:42, end:56, title:'Hai agent không cần train',
      body:'<p><strong>Heuristic</strong>: ưu tiên thắng ngay, sau đó chặn, cộng điểm gần cột giữa và số quân liền kề; thêm nhiễu UCB để khám phá.</p><p><strong>Minimax</strong>: duyệt cây với alpha-beta, cắt nhánh không ảnh hưởng kết quả.</p>',
      tags:['heuristic','minimax','alpha-beta'] },
    { id:'r-usage', start:66, end:114, title:'Cách dùng play / test / train',
      body:'<p><code>play(opponent)</code> để chơi tay; <code>test(player, opponent)</code> đo winrate; <code>train(...)</code> huấn luyện PPO/DQN/A2C.</p><p>Ví dụ benchmark trong README dùng ensemble của 15 agent PPO đã lưu.</p>',
      tags:['API','winrate'] },
    { id:'r-run', start:116, end:127, title:'Cài đặt & chạy',
      body:'<p>Cần <code>gymnasium</code>, <code>stable-baselines3</code>, <code>pygame</code>.</p><p><span class="warn">Lưu ý:</span> <code>play.py</code> và <code>test.py</code> không có <code>if __name__ == "__main__"</code> — chúng chạy ngay khi bị import.</p>',
      tags:['pip','entrypoint'] },
  ],

  'connect4.py': [
    { id:'c4-head', start:1, end:16, title:'Import & hằng số reward',
      body:'<p>Khai báo env Gymnasium, ba thuật toán SB3, Pygame và các agent phụ.</p><p>Reward thực tế: thắng <strong>+10</strong>, chặn <strong>+3</strong>, hòa <strong>+1</strong>, nước sai <strong>−100</strong>. Con số hòa/illegal khác README.</p>',
      tags:['WIN_REWARD','DRAW_REWARD','PENALTY'] },
    { id:'c4-init', start:19, end:41, title:'Không gian quan sát & hành động',
      body:'<p><code>Discrete(7)</code> cho 7 cột. Observation là <code>Box(0..2, shape=(6,7), int8)</code> — đúng bằng lưới bàn cờ.</p><p>Phần còn lại cấu hình Pygame: mỗi ô 100px, bán kính quân 45px, màu đỏ/vàng.</p>',
      tags:['Discrete(7)','Box 6×7','int8'] },
    { id:'c4-step-a', start:42, end:59, title:'step() — nước của agent',
      body:'<p>Nếu cột đầy: trả <code>−100</code>, <code>done=True</code> ngay (không reset). Ngược lại thả quân của agent rồi gọi <code>_check_game_over</code>.</p><p>Nếu ván kết thúc, <code>info["is_success"]</code> chỉ <code>True</code> khi thắng; hòa bị coi là không thành công.</p>',
      tags:['illegal move','is_success'] },
    { id:'c4-step-b', start:61, end:78, title:'step() — dò nước chặn & lượt đối thủ',
      body:'<p>Code đặt <em>tạm</em> quân đối thủ vào đúng ô agent vừa đi để xem nước đó có phải nước chặn không, rồi trả lại quân agent.</p><p>Sau đó đối thủ đi. Nếu đối thủ thắng, agent nhận <code>−reward</code>. Nếu agent vừa chặn và đối thủ không kết thúc được, agent nhận <strong>+3</strong>.</p>',
      tags:['block +3','opponent turn','reward'] },
    { id:'c4-reset', start:80, end:88, title:'reset() — ai đi trước?',
      body:'<p>50% đổi ký hiệu: agent có thể là 1 hoặc 2. Nếu agent là 2, đối thủ được thả một quân trước.</p><p>Đối thủ đi bằng <code>_get_opp_action()</code> — nếu đã gắn agent thì dùng agent đó, không phải random.</p>',
      tags:['random first player','symbols'] },
    { id:'c4-render', start:90, end:116, title:'render() — Pygame & stdout',
      body:'<p><code>time.sleep(1)</code> nằm ở đầu hàm nên mọi lần render đều chậm 1 giây, kể cả khi train/benchmark.</p><p>Chế độ <code>stdout</code> chỉ in mảng; chế độ còn lại vẽ vòng tròn lên cửa sổ Pygame.</p>',
      tags:['pygame','sleep(1)'] },
    { id:'c4-drop', start:118, end:125, title:'_drop_piece & _is_valid_action',
      body:'<p>Thả quân: quét từ hàng 5 lên 0, đặt vào ô trống đầu tiên gặp được (trọng lực).</p><p>Cột hợp lệ khi ô trên cùng <code>grid[0, col]</code> còn trống.</p>',
      tags:['gravity','valid column'] },
    { id:'c4-opp', start:127, end:148, title:'_get_opp_action() — random hoặc epsilon-greedy',
      body:'<p>Chưa gắn đối thủ → chọn random đến khi hợp lệ.</p><p>Đã gắn → với xác suất <code>opp_epsilon</code> đi random, còn lại gọi <code>opp_agent.predict</code>. Vòng lặp lặp lại nếu agent chọn cột đầy.</p>',
      tags:['epsilon-greedy','random opponent'] },
    { id:'c4-check', start:150, end:211, title:'_check_game_over() — hòa trước, thắng sau',
      body:'<p>Kiểm tra <strong>hòa trước</strong>: nếu bàn đầy thì trả <code>DRAW_REWARD</code> ngay, chưa xét thắng.</p><p>Sau đó xét 4 hướng quanh ô vừa đặt bằng các cửa sổ 4 ô: ngang, chéo xuôi, chéo ngược, dọc.</p><p><span class="warn">Hệ quả:</span> nước cuối cùng vừa làm đầy bàn vừa tạo 4 quân sẽ bị tính là <strong>hòa</strong>, không phải thắng.</p>',
      tags:['draw first','4 hướng','edge case'] },
    { id:'c4-set', start:213, end:221, title:'set_opp_agent() & close()',
      body:'<p>Gắn đối thủ và epsilon. <code>close()</code> giải phóng cửa sổ Pygame và đặt <code>screen = None</code>.</p>',
      tags:['API'] },
    { id:'c4-main', start:224, end:303, title:'__main__ — curriculum 16 agent',
      body:'<p>Train <code>agent0</code> 100k bước, lưu vào <code>saved_agents/</code>. Sau đó lặp 15 lần: tạo agent mới, cho đấu với agent trước với <code>epsilon = 0.2</code>, train tiếp 100k bước.</p><p>Đây là self-play kiểu curriculum: đối thủ mạnh dần theo từng vòng.</p>',
      tags:['curriculum','self-play','PPO'] },
  ],

  'heuristic.py': [
    { id:'h-init', start:5, end:9, title:'Khởi tạo',
      body:'<p><code>action_counts</code> bắt đầu bằng 1 cho 7 cột, <code>timesteps</code> đếm số lần predict, <code>c = sqrt(2)</code> là hệ số khám phá UCB1.</p><p>Agent này <strong>có trạng thái</strong>: điểm UCB thay đổi theo lịch sử đã chọn cột.</p>',
      tags:['UCB1','stateful'] },
    { id:'h-turn', start:11, end:21, title:'_determine_turn() — suy ra lượt',
      body:'<p>Đếm số quân 1 và 2: bằng nhau → lượt 1; hơn đúng 1 → lượt 2. Trạng thái lệch sẽ raise <code>ValueError</code>.</p><p>Nhờ vậy agent không cần biết mình là ký hiệu nào, chỉ cần nhìn bàn cờ.</p>',
      tags:['turn inference'] },
    { id:'h-predict', start:23, end:60, title:'predict() — chấm điểm từng cột',
      body:'<p>Mỗi cột hợp lệ bắt đầu với <code>score = 1</code>:</p><p>+ <strong>10000</strong> nếu thắng ngay<br>+ điểm heuristic (giữa bàn, quân liền kề)<br>+ <strong>9000</strong> nếu chặn đối thủ thắng ngay</p><p>Sau đó cộng nhiễu UCB <code>c·sqrt(log(t)/count)</code> rồi lấy argmax. Vì 10000 &gt; 9000, agent luôn ưu tiên thắng trước khi chặn.</p>',
      tags:['10000 thắng','9000 chặn','argmax'] },
    { id:'h-helpers', start:62, end:69, title:'Helper: cột hợp lệ & thả quân',
      body:'<p>Bản sao đơn giản của logic bên <code>connect4.py</code>, thao tác trực tiếp trên numpy array.</p>',
      tags:['helpers'] },
    { id:'h-check', start:71, end:109, title:'_check_game_over() — kiểm tra 4 hướng',
      body:'<p>Chỉ cần kiểm tra quanh ô vừa đặt: ngang, dọc, chéo xuôi, chéo ngược. Khác bản trong env, bản này không xét hòa.</p><p>Dùng để phát hiện “nước này có thắng ngay không”, phục vụ hai mức điểm 10000 và 9000.</p>',
      tags:['win check','4 hướng'] },
    { id:'h-eval', start:111, end:122, title:'_evaluate_board() — điểm vị trí',
      body:'<p>Cộng <code>3 − |3 − col|</code>: cột 3 được +3, cột 0 hoặc 6 được 0. Cộng <code>10 × số quân liền kề</code>.</p><p>Đây là heuristic thủ công, không học từ dữ liệu.</p>',
      tags:['center bias','connections ×10'] },
    { id:'h-conn', start:124, end:152, title:'_count_connections() — đếm quân liền kề',
      body:'<p>Với 4 hướng, đếm quân cùng ký hiệu liên tiếp về hai phía (tối đa 3 mỗi phía), rồi trừ 1 để không tính ô vừa đặt hai lần.</p>',
      tags:['contiguous','4 directions'] },
  ],

  'minimax.py': [
    { id:'m-init', start:6, end:9, title:'Khởi tạo',
      body:'<p>Agent được gắn cố định <code>agent_symb</code> và <code>opp_symb</code> ngay khi tạo. Nếu môi trường đổi ký hiệu mà không tạo lại agent, nó sẽ tính sai phe.</p>',
      tags:['fixed symbols'] },
    { id:'m-helpers', start:11, end:21, title:'Helper: cột hợp lệ & thả quân',
      body:'<p>Giống env: cột hợp lệ khi hàng 0 trống; thả quân bằng cách quét từ dưới lên.</p>',
      tags:['helpers'] },
    { id:'m-win', start:23, end:45, title:'_winning_move() — quét toàn bàn',
      body:'<p>Quét mọi cửa sổ 4 ô theo 4 hướng trên toàn bàn cờ, trả <code>True</code> nếu có 4 quân cùng ký hiệu.</p><p>Hàm này được gọi ở mọi node terminal nên là phần tốn thời gian chính của minimax.</p>',
      tags:['win scan','4 hướng'] },
    { id:'m-terminal', start:47, end:52, title:'_is_terminal_node()',
      body:'<p>Node kết thúc khi một trong hai bên đã có 4 quân, hoặc không còn cột nào trống.</p>',
      tags:['terminal'] },
    { id:'m-search', start:54, end:105, title:'minimax() — alpha-beta',
      body:'<p><strong>Terminal:</strong> thắng <code>+1e11</code>, thua <code>−1e10</code>, còn lại 0.</p><p><strong>Depth = 0 nhưng chưa terminal:</strong> trả 0 — không có hàm đánh giá trung gian.</p><p><span class="warn">Hệ quả (đã đo):</span> vì mọi leaf đều 0, điều kiện cắt <code>alpha &gt;= beta</code> đúng ngay khi cả hai bằng 0. Với bàn trống, agent chỉ duyệt khoảng <strong>76 node</strong> thay vì ~2.800 và gần như luôn chọn cột 0. Bản web thêm <code>leafEval</code> để Minimax+ chơi được; code gốc vẫn hiển thị nguyên vẹn.</p>',
      tags:['1e11 / −1e10','no eval','α-β cắt sớm'] },
    { id:'m-predict', start:107, end:108, title:'predict() — chạy search',
      body:'<p>Gọi minimax với depth mặc định <strong>4</strong>, trả về <code>(cột, điểm)</code>. Trong <code>play.py</code>, <code>player.predict(env.grid)[0]</code> lấy phần cột.</p>',
      tags:['depth 4','(col, value)'] },
    { id:'m-cli', start:111, end:161, title:'Class Connect4 CLI (demo riêng)',
      body:'<p>Đây là một bản Connect4 <strong>độc lập</strong> để chơi bằng dòng lệnh, không được <code>play.py</code> dùng. Nó tạo <code>MinimaxAgent(agent_symb=1, opp_symb=2)</code> và cho người nhập cột 0–6.</p>',
      tags:['CLI demo','unused'] },
  ],

  'ensemble.py': [
    { id:'e-vote', start:1, end:13, title:'EnsembleAgent — bỏ phiếu đa số',
      body:'<p>Mỗi agent trong danh sách dự đoán một cột; <code>np.unique</code> đếm phiếu và chọn cột nhiều phiếu nhất.</p><p><span class="warn">Tie-break:</span> <code>np.unique</code> trả về giá trị đã sắp xếp tăng dần, <code>argmax</code> lấy phần tử đầu tiên → hòa phiếu sẽ chọn <strong>cột nhỏ hơn</strong>.</p><p><span class="warn">Lưu ý API:</span> <code>agent.predict(obs, deterministic)</code> truyền tham số thứ hai theo vị trí, mà SB3 định nghĩa đó là <code>state</code>; với <code>MlpPolicy</code> nó bị bỏ qua nên không lỗi, nhưng cờ deterministic không được bật. Muốn chắc chắn phải viết <code>predict(obs, deterministic=True)</code>.</p>',
      tags:['hard voting','tie → cột nhỏ','predict positional'] },
  ],

  'play.py': [
    { id:'p-col', start:15, end:23, title:'get_col() — đọc click',
      body:'<p>Lấy <code>event.pos[0] // 100</code> làm cột, khớp với <code>cell_size = 100</code> bên env. Đóng cửa sổ → trả <code>None</code>.</p>',
      tags:['pygame mouse','hard-coded 100px'] },
    { id:'p-play', start:26, end:65, title:'play() — vòng chơi',
      body:'<p>Không truyền <code>player</code> → chế độ người chơi + render Pygame. Có <code>player</code> → chế độ stdout cho agent vs agent.</p><p><code>set_opp_agent(opponent, 0)</code> đặt epsilon = 0, nên đối thủ chơi deterministic. Kết quả đọc từ reward: 10 thắng, 1 hòa, còn lại thua.</p>',
      tags:['epsilon 0','reward → kết quả'] },
    { id:'p-entry', start:68, end:79, title:'Entrypoint — chạy ngay khi import',
      body:'<p>Dòng cuối gọi <code>play(opponent=HeuristicAgent())</code> mà không có guard <code>__main__</code>. Import file này từ chỗ khác sẽ mở luôn một ván Pygame.</p>',
      tags:['no __main__ guard'] },
  ],

  'train.py': [
    { id:'t-train', start:18, end:38, title:'train() — chọn thuật toán',
      body:'<p>Map tên <code>PPO/A2C/DQN</code> sang class SB3, tạo env, train với tổng <code>total_timesteps</code>.</p><p><span class="warn">Bug:</span> <code>algos[opp_algo].load("opp_path")</code> đang dùng chuỗi literal <code>"opp_path"</code> thay vì biến <code>opp_path</code>. Truyền đối thủ vào sẽ lỗi file-not-found.</p>',
      tags:['PPO','A2C','DQN','bug: literal'] },
    { id:'t-plug', start:41, end:43, title:'plug_opp() — gắn đối thủ vào VecEnv',
      body:'<p>SB3 bọc env trong <code>DummyVecEnv</code>; <code>env_method("set_opp_agent", ...)</code> gọi hàm này trên env gốc bên trong.</p>',
      tags:['VecEnv','env_method'] },
    { id:'t-curriculum', start:46, end:68, title:'__main__ — curriculum',
      body:'<p>Giống hệt phần main của <code>connect4.py</code>: 16 agent PPO, mỗi agent 100k bước, đối thủ là agent liền trước với epsilon 0.2.</p><p>Chạy trọn vòng này cần khá nhiều thời gian và dung lượng đĩa cho <code>saved_agents/</code>.</p>',
      tags:['16 agents','100k steps'] },
    { id:'t-exp', start:70, end:119, title:'Các thí nghiệm bị comment',
      body:'<p>Gồm: train với heuristic, DQN vs Heuristic, hai agent đối kháng train song song, và các biến thể lưu model. Đây là “phòng thí nghiệm” để tham khảo, không chạy mặc định.</p>',
      tags:['experiments','reference'] },
  ],

  'test.py': [
    { id:'te-test', start:16, end:33, title:'test() — một ván',
      body:'<p>Chạy đúng một episode rồi trả <code>True</code> chỉ khi reward = 10. Hòa và thua đều trả <code>False</code>.</p><p>Tham số <code>mode</code> không được dùng bên trong hàm.</p>',
      tags:['1 episode','win = True'] },
    { id:'te-bench', start:36, end:43, title:'Benchmark — chạy ngay khi import',
      body:'<p>Load 15 model PPO, gộp thành ensemble, chơi 10.000 ván và in winrate. Không có guard <code>__main__</code>, nên chỉ cần import là chạy.</p><p>Thư mục <code>saved_agents/</code> phải tồn tại và có đủ 15 file.</p>',
      tags:['10k games','15 PPO','no guard'] },
  ],
};

const PIPELINE = [
  { label:'play.py · play()', file:'play.py', sec:'p-play' },
  { label:'Connect4.step()', file:'connect4.py', sec:'c4-step-a' },
  { label:'Heuristic.predict()', file:'heuristic.py', sec:'h-predict' },
  { label:'minimax()', file:'minimax.py', sec:'m-search' },
  { label:'PPO.learn()', file:'train.py', sec:'t-train' },
];

/* ------------------------------------------------------------------ */
/* Tiny DOM helpers                                                    */
/* ------------------------------------------------------------------ */

const $ = (s, r = document) => r.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const pace = () => (G.human ? 1 : 0.45);
const wait = (ms) => sleep(ms * pace());

const els = {
  board: $('#board'),
  status: $('#status'),
  reward: $('#reward'),
  banner: $('#banner'),
  steps: $('#steps'),
  players: $('#players'),
  tabs: $('#tabs'),
  code: $('#code'),
  codeWrap: $('#codeWrap'),
  note: $('#note'),
  noteBadge: $('#noteBadge'),
  noteTitle: $('#noteTitle'),
  noteBody: $('#noteBody'),
  noteTags: $('#noteTags'),
  sections: $('#sections'),
  fileMeta: $('#fileMeta'),
  pipeline: $('#pipeline'),
  loadState: $('#loadState'),
};

/* ------------------------------------------------------------------ */
/* Python highlighting                                                 */
/* ------------------------------------------------------------------ */

const PY_TOKEN = /("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#[^\n]*|\b\d+(?:\.\d+)?\b|\b(?:def|class|return|if|elif|else|for|while|in|not|and|or|import|from|as|with|try|except|finally|raise|pass|break|continue|lambda|True|False|None|is|global|yield|assert|del|nonlocal|self)\b)/g;

function tokenize(src) {
  const lines = [[]];
  let last = 0;
  let m;
  PY_TOKEN.lastIndex = 0;

  const push = (txt, cls) => {
    const parts = txt.split('\n');
    for (let i = 0; i < parts.length; i++) {
      if (i > 0) lines.push([]);
      if (parts[i]) lines[lines.length - 1].push([esc(parts[i]), cls]);
    }
  };

  m = PY_TOKEN.exec(src);
  while (m) {
    if (m.index > last) push(src.slice(last, m.index), null);
    const t = m[0];
    let cls = 'tok-kw';
    if (t[0] === '#') cls = 'tok-com';
    else if (t[0] === '"' || t[0] === "'") cls = 'tok-str';
    else if (/^\d/.test(t)) cls = 'tok-num';
    else if (t === 'self') cls = 'tok-self';
    push(t, cls);
    last = m.index + t.length;
    m = PY_TOKEN.exec(src);
  }
  if (last < src.length) push(src.slice(last), null);
  if (lines.length > 1 && lines[lines.length - 1].length === 0 && src.endsWith('\n')) lines.pop();
  return lines;
}

/* ------------------------------------------------------------------ */
/* Explainer state                                                     */
/* ------------------------------------------------------------------ */

const state = {
  sources: {},
  activeFile: 'connect4.py',
  activeSection: null,
  live: false,
  rowEls: {},
};

function sectionFor(file, line) {
  const list = SECTIONS[file] || [];
  return list.find((s) => line >= s.start && line <= s.end) || null;
}

function renderTabs() {
  els.tabs.innerHTML = FILES.map(
    (f) => `<button type="button" class="tab" data-file="${f.name}">
      <span class="ext">${f.ext}</span>${f.name.replace(/\.(py|md)$/, '')}
    </button>`
  ).join('');
  els.tabs.addEventListener('click', (e) => {
    const b = e.target.closest('.tab');
    if (b) {
      $('#follow').checked = false;
      openFile(b.dataset.file);
    }
  });
}

function renderPipeline() {
  els.pipeline.innerHTML = PIPELINE.map(
    (p, i) => `${i ? '<span class="pipe-arrow">→</span>' : ''}
      <button type="button" class="pipe" data-file="${p.file}" data-sec="${p.sec}">${p.label}</button>`
  ).join('');
  els.pipeline.addEventListener('click', (e) => {
    const b = e.target.closest('.pipe');
    if (b) openSection(b.dataset.file, b.dataset.sec);
  });
}

function renderSections(file) {
  const list = SECTIONS[file] || [];
  const lines = state.sources[file] ? state.sources[file].split('\n').length : 0;
  els.fileMeta.textContent = lines ? `${lines} dòng` : '';
  els.sections.innerHTML = list.map(
    (s) => `<button type="button" class="section-chip${state.activeSection === s.id ? ' on' : ''}"
      data-sec="${s.id}"><span class="n">${s.start}</span><span class="t">${s.title}</span></button>`
  ).join('');
}

els.sections.addEventListener('click', (e) => {
  const b = e.target.closest('.section-chip');
  if (!b) return;
  const sec = (SECTIONS[state.activeFile] || []).find((s) => s.id === b.dataset.sec);
  if (sec) {
    $('#follow').checked = false;
    applySection(sec, false);
  }
});

function openFile(name) {
  if (!FILES.some((f) => f.name === name)) return;
  state.activeFile = name;
  state.live = false;

  els.tabs.querySelectorAll('.tab').forEach((t) => {
    t.classList.toggle('on', t.dataset.file === name);
  });

  const src = state.sources[name];
  if (src == null) {
    els.code.innerHTML = `<div class="row"><span class="gutter">!</span><span class="src">Không nạp được ${name}. Hãy chạy qua local server (xem footer).</span></div>`;
    state.rowEls = {};
  } else {
    const plain = name.endsWith('.md');
    const body = plain ? (src.endsWith('\n') ? src.slice(0, -1) : src) : src;
    const lines = plain
      ? body.split('\n').map((line) => [[esc(line), null]])
      : tokenize(body);
    els.code.innerHTML = lines
      .map((parts, i) => {
        const html = parts
          .map(([txt, cls]) => (cls ? `<span class="${cls}">${txt}</span>` : txt))
          .join('');
        return `<div class="row" data-line="${i + 1}"><span class="gutter">${i + 1}</span><span class="src">${html || ' '}</span></div>`;
      })
      .join('');
    state.rowEls = {};
    els.code.querySelectorAll('.row').forEach((el) => {
      state.rowEls[+el.dataset.line] = el;
    });
  }

  const first = (SECTIONS[name] || [])[0];
  state.activeSection = first ? first.id : null;
  renderSections(name);
  if (first) applySection(first, false);
}

function rangeRows(start, end) {
  const out = [];
  for (let i = start; i <= end; i++) if (state.rowEls[i]) out.push(state.rowEls[i]);
  return out;
}

function scrollToRow(row) {
  if (!row) return;
  const box = els.codeWrap.getBoundingClientRect();
  const line = row.getBoundingClientRect();
  els.codeWrap.scrollTop += line.top - box.top - els.codeWrap.clientHeight * 0.38;
}

function applySection(sec, live, noScroll) {
  state.activeSection = sec.id;
  state.live = !!live;

  Object.values(state.rowEls).forEach((r) => {
    r.classList.remove('hl', 'live');
  });
  const rows = rangeRows(sec.start, sec.end);
  rows.forEach((r) => {
    r.classList.add(live ? 'live' : 'hl');
  });
  if (!noScroll) scrollToRow(rows[0]);

  els.noteBadge.textContent = live ? 'LIVE' : 'GIẢI THÍCH';
  els.noteBadge.classList.toggle('live', !!live);
  els.noteTitle.textContent = sec.title;
  els.noteBody.innerHTML = sec.body;
  els.noteTags.innerHTML = (sec.tags || []).map((t) => `<span class="tag">${t}</span>`).join('');

  els.sections.querySelectorAll('.section-chip').forEach((b) => {
    b.classList.toggle('on', b.dataset.sec === sec.id);
  });
}

function openSection(file, id) {
  const sec = (SECTIONS[file] || []).find((s) => s.id === id);
  if (!sec) return;
  $('#follow').checked = false;
  if (state.activeFile !== file) {
    openFile(file);
  }
  applySection(sec, false);
}

function showLive(ev, opts = {}) {
  if (opts.trace !== false) addStep(ev);
  if (!$('#follow').checked) return;
  if (!state.sources[ev.file]) return;
  if (state.activeFile !== ev.file) openFile(ev.file);
  applySection(
    { id: 'live', start: ev.start, end: ev.end, title: ev.title, body: ev.body, tags: ev.tags || [] },
    true
  );
}

/* ------------------------------------------------------------------ */
/* Trace                                                               */
/* ------------------------------------------------------------------ */

const G = {
  gen: 0,
  grid: null,
  turn: 1,
  over: false,
  busy: false,
  agentSymb: 1,
  oppSymb: 2,
  human: true,
  mode: 'human-heuristic',
  leftAgent: null,
  oppAgent: null,
  reward: 0,
  winning: [],
  last: null,
  steps: [],
  result: null,
};

function addStep(ev) {
  G.steps.push({ file: ev.file, start: ev.start, end: ev.end, title: ev.title, body: ev.body, tone: ev.tone || 'dim' });
  if (G.steps.length > 9) G.steps.shift();
  renderSteps();
}

function renderSteps() {
  if (!G.steps.length) {
    els.steps.innerHTML = '<div class="step empty">Chưa có nước đi — bấm một cột để bắt đầu</div>';
    return;
  }
  els.steps.innerHTML = G.steps
    .map(
      (s, i) => `<button type="button" class="step ${s.tone}" data-i="${i}">
        <span class="bar"></span><span>${s.title}</span><code>${s.file}</code>
      </button>`
    )
    .reverse()
    .join('');
}

els.steps.addEventListener('click', (e) => {
  const b = e.target.closest('.step');
  if (!b) return;
  const s = G.steps[+b.dataset.i];
  if (!s) return;
  openFile(s.file);
  applySection({ id: 'trace', start: s.start, end: s.end, title: s.title, body: s.body, tags: [] }, false);
});

/* ------------------------------------------------------------------ */
/* Game rules (mirrors connect4.py)                                    */
/* ------------------------------------------------------------------ */

const ROWS = 6;
const COLS = 7;

const newGrid = () => Array.from({ length: ROWS }, () => Array(COLS).fill(0));

function validCols(g) {
  const v = [];
  for (let c = 0; c < COLS; c++) if (g[0][c] === 0) v.push(c);
  return v;
}

function dropPiece(g, col, player) {
  for (let r = ROWS - 1; r >= 0; r--) {
    if (g[r][col] === 0) {
      g[r][col] = player;
      return r;
    }
  }
  return -1;
}

function boardFull(g) {
  return g[0].every((v) => v !== 0);
}

function findLine(g, row, col) {
  const p = g[row][col];
  if (!p) return [];
  const dirs = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];
  for (const [dr, dc] of dirs) {
    const cells = [[row, col]];
    for (const sign of [1, -1]) {
      let r = row + dr * sign;
      let c = col + dc * sign;
      while (r >= 0 && r < ROWS && c >= 0 && c < COLS && g[r][c] === p) {
        cells.push([r, c]);
        r += dr * sign;
        c += dc * sign;
      }
    }
    if (cells.length >= 4) return cells;
  }
  return [];
}

/* Python checks draw BEFORE win — mirrored on purpose. */
function checkGameOver(g, row, col) {
  if (boardFull(g)) return { draw: true, win: false, cells: [] };
  const cells = findLine(g, row, col);
  return { draw: false, win: cells.length >= 4, cells };
}

/* ------------------------------------------------------------------ */
/* Agents                                                              */
/* ------------------------------------------------------------------ */

class RandomAgent {
  choose(g) {
    const v = validCols(g);
    return v[Math.floor(Math.random() * v.length)];
  }
}

class HeuristicAgent {
  constructor(c = Math.SQRT2) {
    this.c = c;
    this.actionCounts = Array(COLS).fill(1);
    this.timesteps = 0;
  }

  countConnections(g, row, col, player) {
    let total = 0;
    for (const [dr, dc] of [
      [0, 1],
      [1, 0],
      [1, 1],
      [1, -1],
    ]) {
      let count = 1;
      for (let i = 1; i < 4; i++) {
        const r = row + dr * i;
        const c = col + dc * i;
        if (r >= 0 && r < ROWS && c >= 0 && c < COLS && g[r][c] === player) count++;
        else break;
      }
      for (let i = 1; i < 4; i++) {
        const r = row - dr * i;
        const c = col - dc * i;
        if (r >= 0 && r < ROWS && c >= 0 && c < COLS && g[r][c] === player) count++;
        else break;
      }
      total += count - 1;
    }
    return total;
  }

  evaluate(g, row, col, symb) {
    let score = 3 - Math.abs(3 - col);
    score += this.countConnections(g, row, col, symb) * 10;
    return score;
  }

  choose(g, symb) {
    this.timesteps += 1;
    const opp = symb === 1 ? 2 : 1;
    const scores = Array(COLS).fill(-Infinity);

    for (const col of validCols(g)) {
      const mine = g.map((r) => r.slice());
      let row = dropPiece(mine, col, symb);
      let score = 1;
      if (findLine(mine, row, col).length >= 4) score += 10000;
      score += this.evaluate(mine, row, col, symb);

      const theirs = g.map((r) => r.slice());
      row = dropPiece(theirs, col, opp);
      if (findLine(theirs, row, col).length >= 4) score += 9000;

      scores[col] = score;
    }

    for (let c = 0; c < COLS; c++) {
      if (Number.isFinite(scores[c])) {
        scores[c] += this.c * Math.sqrt(Math.log(this.timesteps) / this.actionCounts[c]);
      }
    }

    let best = validCols(g)[0];
    for (let c = 0; c < COLS; c++) if (scores[c] > scores[best]) best = c;
    this.actionCounts[best] += 1;
    return best;
  }
}

class MinimaxAgent {
  constructor(agentSymb, oppSymb, depth = 4, useEval = true) {
    this.agentSymb = agentSymb;
    this.oppSymb = oppSymb;
    this.depth = depth;
    this.useEval = useEval;
  }

  /* Bản gốc trả 0 ở depth 0. Bản web thêm eval để agent chơi được. */
  leafEval(g) {
    const me = this.agentSymb;
    const opp = this.oppSymb;
    const weight = [0, 1, 8, 60];
    let score = 0;

    const scan = (r, c, dr, dc) => {
      let mine = 0;
      let theirs = 0;
      for (let i = 0; i < 4; i++) {
        const v = g[r + dr * i][c + dc * i];
        if (v === me) mine++;
        else if (v === opp) theirs++;
      }
      if (mine && !theirs) score += weight[mine];
      else if (theirs && !mine) score -= weight[theirs] * 1.15;
    };

    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS - 3; c++) scan(r, c, 0, 1);
    for (let r = 0; r < ROWS - 3; r++) for (let c = 0; c < COLS; c++) scan(r, c, 1, 0);
    for (let r = 0; r < ROWS - 3; r++) for (let c = 0; c < COLS - 3; c++) scan(r, c, 1, 1);
    for (let r = 3; r < ROWS; r++) for (let c = 0; c < COLS - 3; c++) scan(r, c, -1, 1);

    for (let r = 0; r < ROWS; r++) {
      if (g[r][3] === me) score += 4;
      else if (g[r][3] === opp) score -= 4;
    }
    return score;
  }

  winningMove(g, piece) {
    for (let r = 0; r < ROWS; r++)
      for (let c = 0; c < COLS - 3; c++)
        if (g[r][c] === piece && g[r][c + 1] === piece && g[r][c + 2] === piece && g[r][c + 3] === piece)
          return true;

    for (let r = 0; r < ROWS - 3; r++)
      for (let c = 0; c < COLS; c++)
        if (g[r][c] === piece && g[r + 1][c] === piece && g[r + 2][c] === piece && g[r + 3][c] === piece)
          return true;

    for (let r = 0; r < ROWS - 3; r++)
      for (let c = 0; c < COLS - 3; c++)
        if (
          g[r][c] === piece &&
          g[r + 1][c + 1] === piece &&
          g[r + 2][c + 2] === piece &&
          g[r + 3][c + 3] === piece
        )
          return true;

    for (let r = 3; r < ROWS; r++)
      for (let c = 0; c < COLS - 3; c++)
        if (
          g[r][c] === piece &&
          g[r - 1][c + 1] === piece &&
          g[r - 2][c + 2] === piece &&
          g[r - 3][c + 3] === piece
        )
          return true;

    return false;
  }

  isTerminal(g) {
    return this.winningMove(g, this.agentSymb) || this.winningMove(g, this.oppSymb) || validCols(g).length === 0;
  }

  minimax(g, depth, alpha, beta, maximizing) {
    const valid = validCols(g);
    const terminal = this.isTerminal(g);

    if (depth === 0 || terminal) {
      if (terminal) {
        if (this.winningMove(g, this.agentSymb)) return [null, 1e11];
        if (this.winningMove(g, this.oppSymb)) return [null, -1e10];
        return [null, 0];
      }
      return [null, this.useEval ? this.leafEval(g) : 0];
    }

    if (maximizing) {
      let value = -Infinity;
      let column = valid[0];
      for (const action of valid) {
        const copy = g.map((r) => r.slice());
        dropPiece(copy, action, this.agentSymb);
        const score = this.minimax(copy, depth - 1, alpha, beta, false)[1];
        if (score > value) {
          value = score;
          column = action;
        }
        alpha = Math.max(alpha, value);
        if (alpha >= beta) break;
      }
      return [column, value];
    }

    let value = Infinity;
    let column = valid[0];
    for (const action of valid) {
      const copy = g.map((r) => r.slice());
      dropPiece(copy, action, this.oppSymb);
      const score = this.minimax(copy, depth - 1, alpha, beta, true)[1];
      if (score < value) {
        value = score;
        column = action;
      }
      beta = Math.min(beta, value);
      if (alpha >= beta) break;
    }
    return [column, value];
  }

  choose(g) {
    return this.minimax(g.map((r) => r.slice()), this.depth, -Infinity, Infinity, true)[0];
  }
}

/* ------------------------------------------------------------------ */
/* Agent profiles (labels + live code mapping)                         */
/* ------------------------------------------------------------------ */

function profile(agent) {
  if (agent instanceof HeuristicAgent)
    return {
      name: 'Heuristic',
      sub: 'heuristic.py',
      file: 'heuristic.py',
      start: 23,
      end: 60,
      title: 'HeuristicAgent.predict() — đang chấm điểm',
      body: '<p>Quét 7 cột: cộng 10000 nếu thắng ngay, 9000 nếu chặn, cộng điểm giữa bàn + quân liền kề, rồi cộng nhiễu UCB và lấy argmax.</p>',
    };
  if (agent instanceof MinimaxAgent)
    return {
      name: 'Minimax+',
      sub: `minimax.py + eval · depth ${agent.depth}`,
      file: 'minimax.py',
      start: 54,
      end: 105,
      title: 'minimax() — đang duyệt cây',
      body: `<p>Duyệt các cột hợp lệ, đệ quy giảm depth, cắt nhánh bằng alpha-beta. Node terminal thắng +1e11 / thua −1e10.</p><p>Bản web thêm <code>leafEval</code> ở depth 0 để chơi được; code gốc trả 0 và gần như chỉ đi cột trái nhất.</p>`,
    };
  return {
    name: 'Random',
    sub: 'connect4.py · random',
    file: 'connect4.py',
    start: 127,
    end: 133,
    title: '_get_random_action() — chọn ngẫu nhiên',
    body: '<p>Lấy ngẫu nhiên một cột hợp lệ. Đây là đối thủ mặc định khi env chưa được gắn agent.</p>',
  };
}

/* ------------------------------------------------------------------ */
/* Board rendering                                                     */
/* ------------------------------------------------------------------ */

function buildBoard() {
  els.board.innerHTML = '';
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'cell';
      b.dataset.col = String(c);
      b.dataset.row = String(r);
      b.setAttribute('aria-label', `Cột ${c + 1}, hàng ${r + 1}`);
      b.innerHTML = '<span class="piece"></span>';
      els.board.appendChild(b);
    }
  }
}

const cellAt = (r, c) => els.board.children[r * COLS + c];

function paint() {
  const wins = new Set(G.winning.map(([r, c]) => `${r},${c}`));
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = cellAt(r, c);
      const v = G.grid[r][c];
      cell.dataset.p = v ? String(v) : '';
      cell.classList.toggle('win', wins.has(`${r},${c}`));
    }
  }
}

function paintDrop(row, col) {
  paint();
  const cell = cellAt(row, col);
  cell.classList.remove('just');
  void cell.offsetWidth;
  cell.classList.add('just');
  setTimeout(() => cell.classList.remove('just'), 420);
}

function clearGhost() {
  els.board.querySelectorAll('.ghost').forEach((c) => {
    c.classList.remove('ghost');
  });
}

function showGhost(col) {
  clearGhost();
  if (!G.human || G.over || G.busy || G.turn !== G.agentSymb) return;
  if (G.grid[0][col] !== 0) return;
  let row = ROWS - 1;
  while (row >= 0 && G.grid[row][col] !== 0) row--;
  if (row >= 0) cellAt(row, col).classList.add('ghost');
}

els.board.addEventListener('pointermove', (e) => {
  const cell = e.target.closest('.cell');
  if (!cell) return clearGhost();
  showGhost(+cell.dataset.col);
});
els.board.addEventListener('pointerleave', clearGhost);

function onColumn(col) {
  if (!G.human || G.over || G.busy || G.turn !== G.agentSymb) return;
  if (G.grid[0][col] !== 0) return;
  clearGhost();
  agentPlay(col);
}
els.board.addEventListener('click', (e) => {
  const cell = e.target.closest('.cell');
  if (cell) onColumn(+cell.dataset.col);
});

/* ------------------------------------------------------------------ */
/* Game flow                                                           */
/* ------------------------------------------------------------------ */

const MODES = {
  'human-heuristic': { human: true, makeOpp: () => new HeuristicAgent() },
  'human-minimax': { human: true, makeOpp: (s, o) => new MinimaxAgent(s, o, 4) },
  'human-random': { human: true, makeOpp: () => new RandomAgent() },
  'spectate-hm': { human: false, makeLeft: () => new HeuristicAgent(), makeOpp: (s, o) => new MinimaxAgent(s, o, 4) },
  'spectate-mh': { human: false, makeLeft: (s, o) => new MinimaxAgent(s, o, 4), makeOpp: () => new HeuristicAgent() },
};

const MODE_NAMES = {
  'human-heuristic': ['Bạn', 'Heuristic'],
  'human-minimax': ['Bạn', 'Minimax+'],
  'human-random': ['Bạn', 'Random'],
  'spectate-hm': ['Heuristic', 'Minimax+'],
  'spectate-mh': ['Minimax+', 'Heuristic'],
};

function renderPlayers() {
  const names = MODE_NAMES[G.mode];
  const left = G.human ? 'Bạn' : names[0];
  const right = names[1];
  const leftSub = G.human ? 'người chơi · symbol ' + G.agentSymb : profile(G.leftAgent).sub + ' · symbol ' + G.agentSymb;
  const rightSub = profile(G.oppAgent).sub + ' · symbol ' + G.oppSymb;

  const leftActive = !G.over && G.turn === G.agentSymb;
  const rightActive = !G.over && G.turn === G.oppSymb;

  els.players.innerHTML = `
    <div class="player ${leftActive ? 'active' : ''}">
      <span class="pawn p${G.agentSymb}"></span>
      <span class="who"><b>${left}</b><small>${leftSub}</small></span>
      <span class="symb">${G.agentSymb}</span>
    </div>
    <div class="player ${rightActive ? 'active' : ''}">
      <span class="pawn p${G.oppSymb}"></span>
      <span class="who"><b>${right}</b><small>${rightSub}</small></span>
      <span class="symb">${G.oppSymb}</span>
    </div>`;
}

function setReward(v) {
  G.reward = v;
  const b = els.reward.querySelector('b');
  b.textContent = v > 0 ? `+${v}` : String(v);
  els.reward.classList.toggle('pos', v > 0);
  els.reward.classList.toggle('neg', v < 0);
}

function updateUI() {
  renderPlayers();
  els.board.style.pointerEvents = !G.human || G.over || G.busy || G.turn !== G.agentSymb ? 'none' : 'auto';

  if (G.over) {
    const leftName = G.human ? 'Bạn' : profile(G.leftAgent).name;
    els.status.innerHTML =
      G.result === 'agent'
        ? `<b>${leftName} thắng.</b> Reward +10 — step() trả về is_success = True.`
        : G.result === 'opponent'
        ? `<b>${profile(G.oppAgent).name} thắng.</b> Reward −10 — agent nhận phần âm của reward.`
        : '<b>Hòa.</b> Reward +1 (theo code) — README ghi 0.';
    return;
  }
  if (G.busy) {
    els.status.innerHTML = '<b>Đang xử lý…</b> agent chọn cột → thả quân → kiểm tra thắng.';
    return;
  }
  if (G.human) {
    els.status.innerHTML =
      G.turn === G.agentSymb
        ? '<b>Lượt của bạn.</b> Bấm một cột để thả quân.'
        : '<b>Đối thủ đang đi…</b>';
  } else {
    els.status.innerHTML = `<b>${G.turn === G.agentSymb ? profile(G.leftAgent).name : profile(G.oppAgent).name}</b> đang suy nghĩ…`;
  }
}

function showBanner(kind, text) {
  els.banner.className = `banner ${kind}`;
  els.banner.textContent = text;
  els.banner.classList.remove('hidden');
}
function hideBanner() {
  els.banner.classList.add('hidden');
}

function newGame() {
  G.gen += 1;
  G.grid = newGrid();
  G.over = false;
  G.busy = false;
  G.turn = 1;
  G.winning = [];
  G.last = null;
  G.steps = [];
  G.result = null;
  G.mode = $('#mode').value;
  G.human = MODES[G.mode].human;

  const first = $('#first').checked;
  G.agentSymb = G.human && !first ? 2 : 1;
  G.oppSymb = 3 - G.agentSymb;

  const cfg = MODES[G.mode];
  G.leftAgent = cfg.makeLeft ? cfg.makeLeft(G.agentSymb, G.oppSymb) : null;
  G.oppAgent = cfg.makeOpp ? cfg.makeOpp(G.oppSymb, G.agentSymb) : null;

  $('#firstWrap').classList.toggle('hidden', !G.human);

  setReward(0);
  hideBanner();
  paint();
  renderSteps();
  updateUI();

  showLive(
    {
      file: 'connect4.py',
      start: 80,
      end: 88,
      title: 'reset() — bắt đầu ván',
      body: `<p>Lưới 6×7 được xóa. Agent nhận symbol <strong>${G.agentSymb}</strong>, đối thủ là <strong>${G.oppSymb}</strong>.</p><p>${
        G.human
          ? first
            ? 'Bạn đi trước.'
            : 'Bạn đi sau: đối thủ thả một quân trước — code gốc random 50%, UI để bạn chọn.'
          : `${profile(G.leftAgent).name} đi trước.`
      }</p>`,
      tone: 'dim',
    },
    { trace: false }
  );

  if (G.human && !first) {
    openingMove(G.gen);
  } else if (!G.human) {
    scheduleLeft(G.gen, 700);
  }
}

async function openingMove(gen) {
  G.busy = true;
  updateUI();
  const p = profile(G.oppAgent);
  showLive({ ...p, title: `${p.name} đi trước (reset)`, tone: 'opp' });
  await wait(420);
  if (gen !== G.gen) return;

  const col = G.oppAgent.choose(G.grid, G.oppSymb);
  const row = dropPiece(G.grid, col, G.oppSymb);
  G.last = { row, col, symb: G.oppSymb };
  paintDrop(row, col);
  updateUI();
  await wait(340);
  if (gen !== G.gen) return;

  G.turn = G.agentSymb;
  G.busy = false;
  updateUI();
}

function scheduleLeft(gen, delay = 560) {
  setTimeout(async () => {
    if (gen !== G.gen || G.over || G.busy || G.turn !== G.agentSymb) return;
    const p = profile(G.leftAgent);
    showLive({ ...p, title: `${p.name} chọn cột`, tone: 'agent' });
    await wait(180);
    if (gen !== G.gen || G.over || G.busy) return;
    const col = G.leftAgent.choose(G.grid, G.agentSymb);
    agentPlay(col, gen);
  }, delay * pace());
}

async function agentPlay(col, gen = G.gen) {
  if (gen !== G.gen || G.over || G.busy) return;
  if (G.turn !== G.agentSymb) return;
  if (G.grid[0][col] !== 0) return;

  G.busy = true;
  updateUI();

  showLive({
    file: 'connect4.py',
    start: 42,
    end: 59,
    title: 'step(action) — nước của agent',
    body: `<p>Kiểm tra cột <strong>${col}</strong> còn trống, thả quân symbol <strong>${G.agentSymb}</strong>, rồi gọi <code>_check_game_over</code>.</p>`,
    tone: 'agent',
  });
  await wait(90);
  if (gen !== G.gen) return;

  const row = dropPiece(G.grid, col, G.agentSymb);
  G.last = { row, col, symb: G.agentSymb };
  paintDrop(row, col);

  showLive(
    {
      file: 'connect4.py',
      start: 118,
      end: 122,
      title: '_drop_piece(action, player)',
      body: `<p>Quét từ hàng 5 lên 0, đặt quân vào ô trống đầu tiên. Cột <strong>${col}</strong> rơi xuống hàng <strong>${row}</strong>.</p>`,
      tone: 'dim',
    },
    { trace: false }
  );
  await wait(310);
  if (gen !== G.gen) return;

  const chk = checkGameOver(G.grid, row, col);
  showLive(
    {
      file: 'connect4.py',
      start: 150,
      end: 211,
      title: '_check_game_over(row, col)',
      body: chk.win
        ? '<p>Phát hiện 4 quân liên tiếp → trả <code>WIN_REWARD = 10</code>, <code>done = True</code>.</p>'
        : chk.draw
        ? '<p>Bàn đầy → trả <code>DRAW_REWARD = 1</code> (code kiểm tra hòa trước khi kiểm tra thắng).</p>'
        : '<p>Chưa có 4 quân và bàn chưa đầy → trả <code>(0, False)</code>.</p>',
      tone: 'check',
    },
    { trace: false }
  );
  await wait(180);
  if (gen !== G.gen) return;

  if (chk.win || chk.draw) {
    setReward(chk.win ? 10 : 1);
    finish(chk.win ? 'agent' : 'draw', chk.cells);
    return;
  }

  /* Block probe — same temporary overwrite as Python step(). */
  showLive(
    {
      file: 'connect4.py',
      start: 61,
      end: 67,
      title: 'Dò nước chặn (block probe)',
      body: '<p>Đặt tạm quân đối thủ vào đúng ô vừa đi để xem nước đó có phải nước chặn không, rồi trả lại quân agent.</p>',
      tone: 'dim',
    },
    { trace: false }
  );
  const saved = G.grid[row][col];
  G.grid[row][col] = G.oppSymb;
  const probe = checkGameOver(G.grid, row, col);
  G.grid[row][col] = saved;
  await wait(160);
  if (gen !== G.gen) return;

  /* Opponent's turn — inside step(). */
  const p = profile(G.oppAgent);
  showLive({ ...p, title: `${p.name} chọn cột`, tone: 'opp' });
  await wait(220);
  if (gen !== G.gen) return;

  const oppCol = G.oppAgent.choose(G.grid, G.oppSymb);
  const oppRow = dropPiece(G.grid, oppCol, G.oppSymb);
  G.last = { row: oppRow, col: oppCol, symb: G.oppSymb };
  paintDrop(oppRow, oppCol);

  showLive(
    {
      file: 'connect4.py',
      start: 68,
      end: 78,
      title: 'Lượt đối thủ trong step()',
      body: `<p>Đối thủ thả cột <strong>${oppCol}</strong> (hàng ${oppRow}). Nếu đối thủ thắng, agent nhận <code>−reward</code>. Nếu agent vừa chặn và đối thủ không kết thúc được, agent nhận <strong>+3</strong>.</p>`,
      tone: 'opp',
    },
    { trace: false }
  );
  await wait(330);
  if (gen !== G.gen) return;

  const oppChk = checkGameOver(G.grid, oppRow, oppCol);
  if (oppChk.win || oppChk.draw) {
    setReward(-(oppChk.win ? 10 : 1));
    finish('opponent', oppChk.cells);
    return;
  }

  if (probe.win) {
    setReward(3);
    showLive({
      file: 'connect4.py',
      start: 76,
      end: 77,
      title: 'BLOCK_REWARD = 3',
      body: '<p>Agent vừa chặn một nước thắng ngay của đối thủ → step() trả <strong>+3</strong>.</p>',
      tone: 'check',
    });
  } else {
    setReward(0);
  }

  G.turn = G.agentSymb;
  G.busy = false;
  updateUI();

  if (!G.human) scheduleLeft(G.gen, 620);
}

function finish(result, cells) {
  G.over = true;
  G.busy = false;
  G.result = result;
  G.winning = cells || [];

  paint();
  updateUI();

  if (result === 'agent') showBanner('win', G.human ? 'Bạn thắng' : `${profile(G.leftAgent).name} thắng`);
  else if (result === 'opponent') showBanner('lose', `${profile(G.oppAgent).name} thắng`);
  else showBanner('draw', 'Hòa');

  showLive({
    file: 'play.py',
    start: 46,
    end: 55,
    title: 'Kết thúc ván — đọc reward',
    body: `<p>Reward cuối = <strong>${G.reward}</strong>. <code>play.py</code> so sánh: 10 → thắng, 1 → hòa, còn lại → thua, rồi đặt caption cửa sổ Pygame.</p>`,
    tone: result === 'agent' ? 'agent' : 'opp',
  });
}

/* ------------------------------------------------------------------ */
/* Load sources & boot                                                 */
/* ------------------------------------------------------------------ */

async function loadSources() {
  const pairs = await Promise.all(
    FILES.map(async (f) => {
      const res = await fetch('../' + f.name, { cache: 'no-store' });
      if (!res.ok) throw new Error(`${f.name}: HTTP ${res.status}`);
      return [f.name, await res.text()];
    })
  );
  state.sources = Object.fromEntries(pairs);
}

function bootControls() {
  $('#newgame').addEventListener('click', newGame);
  $('#mode').addEventListener('change', () => {
    $('#first').checked = true;
    newGame();
  });
  $('#first').addEventListener('change', newGame);
  $('#follow').addEventListener('change', () => {
    if ($('#follow').checked && state.activeSection) {
      const sec = (SECTIONS[state.activeFile] || []).find((s) => s.id === state.activeSection);
      if (sec) applySection(sec, false);
    }
  });
}

async function boot() {
  buildBoard();
  renderTabs();
  renderPipeline();
  bootControls();
  renderSteps();

  try {
    await loadSources();
    els.loadState.textContent = 'source đã nạp · 8 file';
    els.loadState.classList.remove('err');
    openFile('connect4.py');
    els.loadState.textContent = `source đã nạp · ${Object.keys(state.sources).length} file`;
  } catch (err) {
    els.loadState.textContent = 'không nạp được source — cần chạy local server';
    els.loadState.classList.add('err');
    els.status.innerHTML = '<b>UI chạy được nhưng thiếu source.</b> Chạy <code>python3 -m http.server 8765</code> ở thư mục project rồi mở <code>/ui/</code>.';
    openFile('connect4.py');
  }

  G.mode = $('#mode').value;
  G.grid = newGrid();
  G.human = MODES[G.mode].human;
  G.agentSymb = 1;
  G.oppSymb = 2;
  G.oppAgent = MODES[G.mode].makeOpp(2, 1);
  paint();
  newGame();
}

boot();

/* Exposed for debugging / headless checks. */
window.__C4 = { G, RandomAgent, HeuristicAgent, MinimaxAgent, newGrid, dropPiece, validCols, checkGameOver, findLine };
