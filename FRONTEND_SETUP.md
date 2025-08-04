
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

## 🔧 最新修复记录 (2025-01-01)

### 修复的问题
1. **react-scripts版本错误**: package.json中react-scripts版本为`^0.0.0`导致启动失败
2. **TypeScript类型错误**: api.ts中getStocks函数返回类型断言问题
3. **ESLint警告**: 未使用的导入警告

### 解决方案
1. **更新react-scripts版本**:
   ```bash
   # 将package.json中的react-scripts从^0.0.0更新到5.0.1
   npm install
   ```

2. **修复TypeScript类型**:
   ```typescript
   // 在api.ts中使用正确的类型断言
   return response as unknown as { stocks: StockInfo[]; total: number };
   ```

3. **清理未使用导入**:
   - 移除ChartControls.tsx中未使用的Checkbox导入
   - 移除Header.tsx中未使用的Title导入

### 技术栈更新
- **React**: 18.2.0
- **TypeScript**: 4.9.5
- **Ant Design**: 5.12.8
- **Plotly.js**: 2.27.1
- **Axios**: 1.6.2

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

### 如果遇到"react-scripts: not found"错误：
```bash
# 检查package.json中react-scripts版本是否正确
cat package.json | grep react-scripts
# 如果版本为^0.0.0或其他无效版本，手动修复：
npm install react-scripts@5.0.1 --save
```

### 如果npm install失败：
```bash
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### 如果遇到TypeScript编译错误：
```bash
# 检查TypeScript配置
npx tsc --noEmit
# 如果有类型错误，检查api.ts中的类型断言
```

### 如果端口冲突：
```bash
PORT=3001 npm start
```

### 如果API连接失败：
- 确保后端服务运行在端口8000
- 检查防火墙设置
- 查看浏览器控制台错误信息

### 如果出现Plotly.js源码映射警告：
这是正常现象，不影响功能。警告信息如下：
```
Failed to parse source map from 'plotly.js/dist/maplibre-gl-unminified.js.map'
```
可以忽略此警告，或在开发环境中禁用源码映射。
