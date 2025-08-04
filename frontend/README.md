# 股票分析系统前端

这是股票分析系统的React前端应用，提供基于春节时间锚点的股票季节性分析功能。

## 功能特性

- 🔍 **股票搜索**: 支持股票代码和名称的智能搜索
- 📊 **春节分析图表**: 基于Plotly的交互式图表展示
- 📱 **响应式设计**: 支持桌面和移动设备
- 🎨 **现代UI**: 基于Ant Design的美观界面
- 📈 **多种图表类型**: 叠加图、对比图、模式图
- 💾 **图表导出**: 支持PNG、SVG、HTML格式导出

## 技术栈

- **React 18** - 前端框架
- **TypeScript** - 类型安全
- **Ant Design** - UI组件库
- **Plotly.js** - 图表可视化
- **Axios** - HTTP客户端
- **Lodash** - 工具函数库

## 开发环境设置

### 前置要求

- Node.js >= 16.0.0
- npm >= 8.0.0

### 安装依赖

```bash
cd frontend
npm install
```

### 启动开发服务器

```bash
npm start
```

应用将在 http://localhost:3000 启动

### 构建生产版本

```bash
npm run build
```

构建文件将输出到 `build/` 目录

## 项目结构

```
frontend/
├── public/                 # 静态资源
│   ├── index.html         # HTML模板
│   └── manifest.json      # PWA配置
├── src/
│   ├── components/        # React组件
│   │   ├── Header.tsx     # 顶部导航
│   │   ├── MainContent.tsx # 主内容区
│   │   ├── StockSearch.tsx # 股票搜索
│   │   ├── ChartControls.tsx # 图表控制
│   │   └── SpringFestivalChart.tsx # 春节图表
│   ├── services/          # API服务
│   │   └── api.ts         # API接口定义
│   ├── App.tsx            # 主应用组件
│   ├── index.tsx          # 应用入口
│   └── index.css          # 全局样式
├── package.json           # 项目配置
└── tsconfig.json          # TypeScript配置
```

## 组件说明

### Header
顶部导航栏，包含应用标题和菜单项。

### MainContent
主内容区域，管理股票搜索和图表显示的状态。

### StockSearch
股票搜索组件，支持：
- 自动完成搜索
- 防抖优化
- 股票信息显示

### ChartControls
图表控制面板，包含：
- 年份选择器
- 图表类型选择
- 显示选项开关

### SpringFestivalChart
春节分析图表组件，功能：
- 基于Plotly的交互式图表
- 图表导出功能
- 加载和错误状态处理

## API集成

前端通过代理连接到后端API（默认端口8000）：

```json
{
  "proxy": "http://localhost:8000"
}
```

主要API端点：
- `GET /api/v1/visualization/sample` - 获取示例图表
- `POST /api/v1/visualization/spring-festival-chart` - 获取春节分析图表
- `POST /api/v1/visualization/export` - 导出图表

## 样式定制

应用使用Ant Design主题系统，可以通过修改 `src/index.css` 来定制样式：

```css
/* 主题色彩 */
.ant-layout-header {
  background: #001529;
}

/* 响应式断点 */
@media (max-width: 768px) {
  .main-content {
    padding: 16px;
  }
}
```

## 部署

### 开发环境
```bash
npm start
```

### 生产环境
```bash
npm run build
npm install -g serve
serve -s build -l 3000
```

### Docker部署
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 浏览器支持

- Chrome >= 88
- Firefox >= 85
- Safari >= 14
- Edge >= 88

## 开发指南

### 添加新组件
1. 在 `src/components/` 创建新的 `.tsx` 文件
2. 使用TypeScript接口定义props
3. 遵循Ant Design设计规范
4. 添加响应式支持

### API集成
1. 在 `src/services/api.ts` 添加新的API函数
2. 定义TypeScript接口
3. 添加错误处理
4. 使用React hooks管理状态

### 样式开发
1. 优先使用Ant Design组件样式
2. 自定义样式写在 `src/index.css`
3. 使用CSS变量保持一致性
4. 确保移动端适配

## 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 使用不同端口启动
   PORT=3001 npm start
   ```

2. **API连接失败**
   - 确保后端服务运行在端口8000
   - 检查代理配置是否正确

3. **图表不显示**
   - 检查Plotly.js是否正确安装
   - 确认图表数据格式正确

4. **构建失败**
   ```bash
   # 清理缓存重新安装
   rm -rf node_modules package-lock.json
   npm install
   ```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

MIT License