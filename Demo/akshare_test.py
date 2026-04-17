import matplotlib.font_manager
fonts = matplotlib.font_manager.fontManager.ttflist
chinese_fonts = [f.name for f in fonts if any(c in f.name.lower() for c in ['hei', 'yahei', 'song', 'sim', 'kai', 'fang'])]
print("可用的中文字体:", chinese_fonts)