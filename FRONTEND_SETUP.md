
# 前端设置说明

## 1. 安装Node.js和npm
确保安装了Node.js 16或更高版本：
```bash
node --version  # 应该显示 v16.x.x 或更高
npm --version   # 应该显示 8.x.x 或更高
```

## 2. 安装前端依赖
```bash
cd frontend
npm install
```

## 3. 启动开发服务器
```bash
npm start
```
前端应用将在 http://localhost:3000 启动

## 4. 启动后端服务器
在另一个终端中：
```bash
python start_server.py
```
后端API将在 http://localhost:8000 启动

## 5. 访问应用
打开浏览器访问 http://localhost:3000

## 6. 功能测试
- 在搜索框中输入股票代码或名称（如：000001 或 平安银行）
- 选择要分析的年份
- 查看生成的春节分析图表
- 尝试导出图表功能

## 故障排除

### 如果npm install失败：
```bash
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### 如果端口冲突：
```bash
PORT=3001 npm start
```

### 如果API连接失败：
- 确保后端服务运行在端口8000
- 检查防火墙设置
- 查看浏览器控制台错误信息
