# SEO 优化指南

## 已实现的 SEO 功能

### 1. 基础 SEO 配置

- ✅ **robots.txt**: 允许搜索引擎抓取所有页面
- ✅ **sitemap.xml**: Hugo 自动生成，提交到 Google Search Console
- ✅ **canonical URLs**: 防止重复内容问题
- ✅ **meta keywords**: 针对每篇文章的关键词
- ✅ **meta description**: 自动从文章摘要生成（160字符限制）

### 2. Open Graph (Facebook/LinkedIn)

每篇文章自动生成：
- `og:type`: article
- `og:title`: 文章标题
- `og:description`: 文章摘要
- `og:url`: 文章链接
- `og:image`: 文章封面图或默认图片
- `article:published_time`: 发布时间
- `article:modified_time`: 修改时间
- `article:author`: 作者
- `article:tag`: 文章标签

### 3. Twitter Cards

- `twitter:card`: summary_large_image
- `twitter:title`: 文章标题
- `twitter:description`: 文章摘要
- `twitter:image`: 文章封面图

### 4. Schema.org 结构化数据

每篇文章包含 JSON-LD 格式的结构化数据：
- BlogPosting 类型
- 作者信息
- 发布/修改时间
- 文章描述
- 图片

## 文章 Front Matter 最佳实践

```yaml
---
title: "文章标题"
date: 2025-10-30T10:00:00+08:00
draft: false
author: "Tang Yong"
tags: ["autonomous driving", "AI", "robotics"]
categories: ["Research"]
description: "简短的文章描述，会用于 SEO meta description"
cover:
  image: "/images/article-cover.png"
  alt: "封面图描述"
  caption: "图片说明"
math: true  # 如果文章包含数学公式
ShowToc: true  # 显示目录
TocOpen: false  # 默认折叠目录
---
```

## Google Search Console 设置

1. 访问 [Google Search Console](https://search.google.com/search-console)
2. 添加网站: `https://qwagrox.github.io`
3. 验证所有权（使用 HTML 标签方法）：
   - 复制验证码
   - 添加到 `hugo.yml` 的 `params.analytics.google.SiteVerificationTag`
4. 提交 sitemap: `https://qwagrox.github.io/sitemap.xml`

## 性能优化

### 已实现：
- ✅ 图片懒加载
- ✅ CSS/JS 压缩（minify）
- ✅ 缓存策略（通过 _headers 文件）
- ✅ 响应式图片

### 建议：
- 📸 使用 WebP 格式图片
- 🗜️ 压缩图片（推荐工具：TinyPNG, ImageOptim）
- 📦 使用 CDN 加速静态资源

## 内容优化建议

### 标题优化
- 使用清晰、描述性的标题
- 包含主要关键词
- 长度控制在 60 字符以内（搜索结果显示限制）

### 摘要优化
- 每篇文章写 2-3 句话的摘要
- 包含关键词
- 长度 150-160 字符（meta description 限制）

### 关键词策略
- 每篇文章 3-5 个标签
- 使用具体、相关的关键词
- 避免关键词堆砌

### 内部链接
- 在文章中链接到相关文章
- 使用描述性的锚文本
- 建立内容层次结构

## 监控和分析

### Google Analytics（可选）
在 `hugo.yml` 中添加：
```yaml
googleAnalytics: "G-XXXXXXXXXX"
```

### 性能监控工具
- [PageSpeed Insights](https://pagespeed.web.dev/)
- [GTmetrix](https://gtmetrix.com/)
- [WebPageTest](https://www.webpagetest.org/)

## 社交媒体优化

### 创建 Open Graph 图片
- 尺寸: 1200x630px
- 格式: PNG 或 JPG
- 文件大小: < 1MB
- 保存到: `static/images/og-image.png`

### 测试工具
- [Facebook Sharing Debugger](https://developers.facebook.com/tools/debug/)
- [Twitter Card Validator](https://cards-dev.twitter.com/validator)
- [LinkedIn Post Inspector](https://www.linkedin.com/post-inspector/)

## 定期维护

- [ ] 每月检查 Google Search Console 的错误
- [ ] 更新过时的内容
- [ ] 修复失效的链接
- [ ] 监控页面加载速度
- [ ] 分析热门内容，创作相关主题

## 进阶优化

### 1. 添加面包屑导航
已启用：`ShowBreadCrumbs: true`

### 2. 添加阅读时间估算
已启用：`ShowReadingTime: true`

### 3. 添加字数统计
已启用：`ShowWordCount: true`

### 4. 启用 RSS
已配置，订阅地址：`https://qwagrox.github.io/index.xml`

## 常见问题

### Q: 为什么我的网站没有出现在 Google 搜索结果中？
A: 新网站需要 1-4 周才能被索引。提交 sitemap 到 Google Search Console 可以加速这个过程。

### Q: 如何提高搜索排名？
A: 
1. 定期发布高质量、原创内容
2. 优化关键词和 meta 标签
3. 建立外部链接（其他网站链接到你的博客）
4. 提高页面加载速度
5. 确保移动端友好

### Q: 如何测试 SEO 效果？
A: 使用以下工具：
- Google Search Console（索引状态、搜索查询）
- Google Analytics（流量来源、用户行为）
- Ahrefs / SEMrush（关键词排名、竞争分析）

