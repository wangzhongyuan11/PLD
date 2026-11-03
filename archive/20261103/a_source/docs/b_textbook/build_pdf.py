"""Build the independent B-group FPGA textbook PDF."""
from __future__ import annotations
import html, re
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, PageBreak,
    PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = Path(__file__).with_name('B组教材_正文.md')
OUT = ROOT / 'output/pdf/B组FPGA加速器从零到独立实现教材.pdf'
pdfmetrics.registerFont(TTFont('CJK', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('CJKB', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('Mono', 'C:/Windows/Fonts/consola.ttf'))
pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJKB', italic='CJK', boldItalic='CJKB')
NAVY=colors.HexColor('#102D49'); BLUE=colors.HexColor('#155C94')
TEAL=colors.HexColor('#148782'); PALE=colors.HexColor('#EAF4F7')
LIGHT=colors.HexColor('#F2F6FA'); TEXT=colors.HexColor('#1B2B38')
MUTED=colors.HexColor('#5B7184'); RULE=colors.HexColor('#CEDCE6')
PW,PH=A4; CW=PW-40*mm

def sty(name, **kw):
    options=dict(fontName='CJK',textColor=TEXT,wordWrap='CJK',allowWidows=0,allowOrphans=0)
    options.update(kw)
    return ParagraphStyle(name,**options)
ST={
 'tag':sty('tag',fontName='CJKB',fontSize=11,leading=18,textColor=TEAL,spaceAfter=11),
 'title':sty('title',fontName='CJKB',fontSize=28,leading=42,textColor=NAVY,spaceAfter=14),
 'sub':sty('sub',fontSize=13.5,leading=24,textColor=BLUE,spaceAfter=25),
 'cover':sty('cover',fontSize=10.5,leading=18,textColor=MUTED,spaceAfter=9),
 'h1':sty('h1',fontName='CJKB',fontSize=19,leading=29,textColor=NAVY,spaceBefore=4,spaceAfter=13,keepWithNext=True),
 'h2':sty('h2',fontName='CJKB',fontSize=13.2,leading=22,textColor=BLUE,spaceBefore=14,spaceAfter=7,keepWithNext=True),
 'h3':sty('h3',fontName='CJKB',fontSize=11,leading=19,textColor=TEAL,spaceBefore=9,spaceAfter=5,keepWithNext=True),
 'body':sty('body',fontSize=10.7,leading=19.2,spaceAfter=9.5),
 'bullet':sty('bullet',fontSize=10.4,leading=18,leftIndent=13,firstLineIndent=-9,spaceAfter=5.5),
 'call':sty('call',fontSize=10.4,leading=18.2,textColor=NAVY),
 'caption':sty('caption',fontSize=8.2,leading=13,textColor=MUTED,alignment=1,spaceBefore=5,spaceAfter=10),
 'cell':sty('cell',fontSize=8.5,leading=13.5),
 'cellh':sty('cellh',fontName='CJKB',fontSize=8.5,leading=13.5,textColor=colors.white),
}
def markup(raw):
    chunks=re.split(r'(\*\*[^*]+\*\*|`[^`]+`)',raw)
    out=[]
    for s in chunks:
        if s.startswith('**') and s.endswith('**'): out.append('<b>'+html.escape(s[2:-2])+'</b>')
        elif s.startswith('`') and s.endswith('`'):
            value=s[1:-1]
            if value.startswith(('http://','https://')):
                safe=html.escape(value,quote=True)
                out.append('<link href="'+safe+'" color="#155C94">'+safe+'</link>')
            else:
                out.append('<font color="#155C94">'+html.escape(value)+'</font>')
        else: out.append(html.escape(s))
    return ''.join(out)
def P(s,k='body'): return Paragraph(markup(s),ST[k])
def callout(label,body):
    p=Paragraph('<font color="#148782"><b>'+html.escape(label)+'</b></font>  '+markup(body),ST['call'])
    t=Table([[p]],colWidths=[CW],hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('BOX',(0,0),(-1,-1),.45,RULE),
        ('LINEBEFORE',(0,0),(0,-1),3.5,TEAL),('LEFTPADDING',(0,0),(-1,-1),10),
        ('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),8),
        ('BOTTOMPADDING',(0,0),(-1,-1),8)]))
    return t
def make_table(lines):
    rows=[[c.strip() for c in l.strip().strip('|').split('|')] for l in lines
          if not re.match(r'^\|?\s*:?-{3,}',l)]
    n=len(rows[0]); assert all(len(r)==n for r in rows),rows
    data=[[P(c,'cellh' if i==0 else 'cell') for c in row] for i,row in enumerate(rows)]
    t=Table(data,colWidths=[CW/n]*n,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
        ('GRID',(0,0),(-1,-1),.35,RULE),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    return t
def make_code(lines,lang):
    for line in lines:
        if len(line)>96: print('WARNING long code line',line[:120])
    pre=Preformatted('\n'.join(lines).rstrip(),ParagraphStyle('code',
        fontName='Mono',fontSize=7.3,leading=11,textColor=NAVY))
    label=Paragraph(html.escape(lang or 'text'),sty('codelabel',fontName='CJKB',
        fontSize=7.2,leading=10,textColor=TEAL))
    t=Table([[label],[pre]],colWidths=[CW],hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),
        ('LINEBEFORE',(0,0),(0,-1),2,BLUE),('LEFTPADDING',(0,0),(-1,-1),9),
        ('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,0),7),
        ('BOTTOMPADDING',(0,0),(-1,0),1),('TOPPADDING',(0,1),(-1,1),0),
        ('BOTTOMPADDING',(0,1),(-1,1),8)]))
    return t
def architecture_diagram():
    d=Drawing(CW,102)
    labels=[('C 预处理','RGB / INT8'),('DDR / DMA','输入与参数'),
            ('B 卷积引擎','MAC / Q24'),('B 非卷积','Pool / Skip'),('C 展示','掩膜回映')]
    box_w=85; gap=(CW-5*box_w)/4
    for i,(a,b) in enumerate(labels):
        x=i*(box_w+gap)
        d.add(Rect(x,25,box_w,52,rx=5,ry=5,strokeColor=RULE,
                   fillColor=PALE if i in (0,4) else LIGHT))
        d.add(String(x+box_w/2,56,a,textAnchor='middle',fontName='CJKB',
                     fontSize=9,fillColor=NAVY))
        d.add(String(x+box_w/2,38,b,textAnchor='middle',fontName='CJK',
                     fontSize=7.3,fillColor=BLUE))
        if i<4:
            x0=x+box_w+2; x1=x+box_w+gap-3
            d.add(Line(x0,51,x1,51,strokeColor=TEAL,strokeWidth=1.3))
            d.add(Polygon([x1,51,x1-5,54,x1-5,48],fillColor=TEAL,strokeColor=TEAL))
    return [d,P('图 1  教学用数据流：真实工程中的物理接口、缓冲与寄存器由 B/C 冻结。','caption')]
def image_pair():
    base=ROOT/'data/raw/isic2016/train'
    im=base/'images/ISIC_0000000.jpg'; mask=base/'masks/ISIC_0000000_Segmentation.png'
    if not im.exists() or not mask.exists():
        return [callout('示例图片','本机缺少 ISIC_0000000 图像对；读者可从官方训练 ZIP 提取后自行查看。')]
    def fit(path):
        with PILImage.open(path) as src: w,h=src.size
        f=min(70*mm/w,43*mm/h)
        return Image(str(path),width=w*f,height=h*f)
    t=Table([[fit(im),fit(mask)]],colWidths=[CW/2]*2,hAlign='CENTER')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('BACKGROUND',(0,0),(-1,-1),LIGHT),
        ('BOX',(0,0),(-1,-1),.45,RULE),('TOPPADDING',(0,0),(-1,-1),9),
        ('BOTTOMPADDING',(0,0),(-1,-1),9)]))
    return [t,P('图 1  官方训练图 ISIC_0000000（左）及对应二值掩膜（右）。来源：ISIC 2016 Task 1，CC0。','caption')]
class BookDoc(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(path,pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,
            topMargin=21*mm,bottomMargin=20*mm,title='B组FPGA加速器从零到独立实现教材',
            author='嵌赛项目技术教材',subject='PYNQ-Z2 Tiny U-Net INT8 FPGA 加速')
        frame=Frame(self.leftMargin,self.bottomMargin,self.width,self.height,
            leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='normal',frames=[frame],onPage=self.draw_page))
        self.chapter='导读'; self.heading_seq=0
    def beforeDocument(self):
        self.heading_seq=0
        self.chapter='导读'
    def draw_page(self,canvas,doc):
        canvas.saveState()
        if doc.page==1:
            canvas.setFillColor(NAVY); canvas.rect(0,PH-13*mm,PW,13*mm,fill=1,stroke=0)
            canvas.setFillColor(TEAL); canvas.rect(0,PH-15*mm,PW,2*mm,fill=1,stroke=0)
            canvas.setFillColor(LIGHT); canvas.rect(0,0,PW,14*mm,fill=1,stroke=0)
        else:
            canvas.setStrokeColor(RULE); canvas.line(20*mm,PH-15*mm,PW-20*mm,PH-15*mm)
            canvas.setFont('CJK',8); canvas.setFillColor(MUTED)
            canvas.drawString(20*mm,PH-11.5*mm,'B组 FPGA 加速器教材')
            canvas.drawRightString(PW-20*mm,PH-11.5*mm,'从数值合同到板端验收')
            canvas.line(20*mm,15*mm,PW-20*mm,15*mm)
            canvas.drawString(20*mm,10*mm,'PYNQ-Z2 | RTL | INT8 | AXI')
            canvas.drawRightString(PW-20*mm,10*mm,str(doc.page))
        canvas.restoreState()
    def afterFlowable(self,f):
        if not isinstance(f,Paragraph) or f.style.name not in ('h1','h2'): return
        title=f.getPlainText(); level=0 if f.style.name=='h1' else 1
        if title=='学习路线与目录': return
        if level==0: self.chapter=title
        self.heading_seq+=1; key=f'sec-{self.heading_seq}'; self.canv.bookmarkPage(key)
        self.notify('TOCEntry',(level,title,self.page,key))
        if level==0: self.canv.addOutlineEntry(title,key,0,False)
def parse():
    lines=MANUSCRIPT.read_text(encoding='utf-8').splitlines(); story=[]; buf=[]; i=0
    def flush():
        if buf: story.append(P(' '.join(s.strip() for s in buf))); buf.clear()
    while i<len(lines):
        l=lines[i].rstrip()
        if not l: flush()
        elif l=='<cover>':
            flush(); story.extend([Spacer(1,37*mm),P('从零开始 · 走向独立闭环','tag'),
                P('B 组 FPGA 加速器教材','title'),
                P('PYNQ-Z2 · 整数 CNN · RTL · AXI/DMA · 板端验证','sub'),
                callout('学习目标','读完并完成练习后，你应能独立读懂 A 组交付，设计、实现、验证和上板一套数值可信的 B 组 FPGA 加速器，并与 C 组完成接口交接。'),
                Spacer(1,14*mm),
                P('以 2026-09-25 冻结的 A 组基线接口为案例。面向没有数字电路或 FPGA 基础的读者。','cover'),
                P('先懂数值和时序，再手算资源，最后逐关完成仿真、实现和板端验收。','cover'),
                Spacer(1,28*mm),P('版本 1.0  |  竞赛学习用途  |  不宣称已完成 B 组硬件','cover'),PageBreak()])
        elif l=='<toc>':
            flush(); story.append(P('学习路线与目录','h1'))
            story.append(P('建议顺序：先做概念与纸笔练习，再实现局部 RTL，最后完成系统联调。每章的“自检”用于验证理解。'))
            toc=TableOfContents(); toc.levelStyles=[
                ParagraphStyle('toc0',fontName='CJKB',fontSize=10.4,leading=19,textColor=NAVY),
                ParagraphStyle('toc1',fontName='CJK',fontSize=8.9,leading=15,textColor=MUTED,leftIndent=15)]
            story.extend([toc,PageBreak()])
        elif l=='<pagebreak>': flush(); story.append(PageBreak())
        elif l=='<imagepair>': flush(); story.extend(image_pair())
        elif l=='<architecture>': flush(); story.extend(architecture_diagram())
        elif l.startswith('# '):
            flush()
            while story and isinstance(story[-1],Spacer): story.pop()
            if story and not isinstance(story[-1],PageBreak):
                story.extend([CondPageBreak(110*mm),Spacer(1,8*mm)])
            story.append(P(l[2:],'h1'))
        elif l.startswith('## '): flush(); story.append(P(l[3:],'h2'))
        elif l.startswith('### '): flush(); story.append(P(l[4:],'h3'))
        elif l.startswith('! '):
            flush(); label,_,body=l[2:].partition('：')
            story.extend([callout(label,body),Spacer(1,7)])
        elif l.startswith('- '): flush(); story.append(P('• '+l[2:],'bullet'))
        elif l.startswith('```'):
            flush(); lang=l[3:].strip(); code=[]; i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]); i+=1
            story.extend([make_code(code,lang),Spacer(1,7)])
        elif l.startswith('|'):
            flush(); tlines=[]
            while i<len(lines) and lines[i].startswith('|'):
                tlines.append(lines[i]); i+=1
            story.extend([make_table(tlines),Spacer(1,8)]); continue
        else: buf.append(l)
        i+=1
    flush(); return story
if __name__=='__main__':
    OUT.parent.mkdir(parents=True,exist_ok=True)
    BookDoc(str(OUT)).multiBuild(parse())
    print(OUT); print('size_bytes=',OUT.stat().st_size)
