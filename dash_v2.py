import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="🌍 기후기술 수준조사 통계정보 대시보드",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)
# CSS 스타일
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #00c9ff 0%, #92fe9d 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00c9ff;
        margin-bottom: 1rem;
    }
    .tech-summary-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .story-box {
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .insight-highlight {
        background: #fefce8;
        border: 1px solid #facc15;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .country-performance {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 기술 설명 데이터 (예시, 실제 데이터로 추후 교체 예정)
TECH_DESCRIPTIONS = {
    "원자력발전": {
        "description": "차세대 원자로 기술을 통한 안전하고 효율적인 전력 생산 기술",
        "korea_status": "한국은 APR1400 상용화로 추격 그룹에 위치하여 세계 4위 수준의 기술력을 확보",
        "global_trend": "미국의 SMR 기술개발이 급상승하고 있는 가운데, 중국의 대용량 원전 건설이 활발히 진행"
    },
    "태양광": {
        "description": "태양 에너지를 전기 에너지로 변환하는 광전지 기술",
        "korea_status": "한국은 고효율 실리콘 셀 기술로 추격 그룹에 위치하여 세계 3위 수준의 기술력을 확보",
        "global_trend": "중국의 제조 기술이 급상승하고 있는 가운데, 유럽의 페로브스카이트 차세대 기술 개발이 활발"
    },
    "수자원관리": {
        "description": "기후변화에 따른 물 부족 및 홍수 등 수자원 문제 해결 기술",
        "korea_status": "한국은 해수담수화 및 스마트 워터 기술로 추격 그룹에 위치하여 아시아 최고 수준의 기술력을 확보",
        "global_trend": "EU의 순환경제 기반 물 재활용 기술이 급상승하고 있는 가운데, 이스라엘-호주의 스마트 워터 기술이 확산"
    }
}

# 데이터 로딩 함수
@st.cache_data(ttl=3600)
def load_climate_tech_data():
    """기후기술 데이터 로드 및 전처리"""
    try:
        # Excel 파일 읽기
        df = pd.read_excel('tracker2020.xlsx', sheet_name=0)

        # 컬럼명 정리
        column_mapping = {
            '세부기술': 'tech_detail',
            '중분류': 'tech_category',
            '감축/적응': 'type',
            '최고 기술 보유국': 'leading_country',
            '한국-기술 수준 (%)': 'kr_tech_level',
            '한국-기술 격차 (년)': 'kr_tech_gap',
            '한국-기술 수준 그룹': 'kr_tech_group',
            '중국-기술 수준 (%)': 'cn_tech_level',
            '중국-기술 격차 (년)': 'cn_tech_gap',
            '일본-기술 수준 (%)': 'jp_tech_level',
            '일본-기술 격차 (년)': 'jp_tech_gap',
            '미국-기술 수준 (%)': 'us_tech_level',
            '미국-기술 격차 (년)': 'us_tech_gap',
            'EU-기술 수준 (%)': 'eu_tech_level',
            'EU-기술 격차 (년)': 'eu_tech_gap',
            '한국-연구 개발 활동 경향': 'kr_rd_trend',
            '한국-기초 연구 역량(점)': 'kr_basic_research',
            '한국-응용 개발 연구 역량(점)': 'kr_applied_research',
            '중국-연구 개발 활동 경향': 'cn_rd_trend',
            '중국-기초 연구 역량(점)': 'cn_basic_research',
            '중국-응용 개발 연구 역량(점)': 'cn_applied_research',
            '일본-연구 개발 활동 경향': 'jp_rd_trend',
            '일본-기초 연구 역량(점)': 'jp_basic_research',
            '일본-응용 개발 연구 역량(점)': 'jp_applied_research',
            '미국-연구 개발 활동 경향': 'us_rd_trend',
            '미국-기초 연구 역량(점)': 'us_basic_research',
            '미국-응용 개발 연구 역량(점)': 'us_applied_research',
            'EU-연구 개발 활동 경향': 'eu_rd_trend',
            'EU-기초 연구 역량(점)': 'eu_basic_research',
            'EU-응용 개발 연구 역량(점)': 'eu_applied_research'
        }

        df = df.rename(columns=column_mapping)

        # 숫자 컬럼 변환
        numeric_cols = [col for col in df.columns if 'tech_level' in col or 'tech_gap' in col or 'research' in col]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 중분류별 데이터 집계 (평균값 사용)
        category_data = df.groupby('tech_category').agg({
            'type': 'first',
            'kr_tech_level': 'mean',
            'kr_tech_gap': 'mean',
            'kr_tech_group': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A',
            'cn_tech_level': 'mean',
            'cn_tech_gap': 'mean',
            'jp_tech_level': 'mean',
            'jp_tech_gap': 'mean',
            'us_tech_level': 'mean',
            'us_tech_gap': 'mean',
            'eu_tech_level': 'mean',
            'eu_tech_gap': 'mean',
            'kr_rd_trend': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A',
            'kr_basic_research': 'mean',
            'kr_applied_research': 'mean',
            'cn_basic_research': 'mean',
            'cn_applied_research': 'mean',
            'jp_basic_research': 'mean',
            'jp_applied_research': 'mean',
            'us_basic_research': 'mean',
            'us_applied_research': 'mean',
            'eu_basic_research': 'mean',
            'eu_applied_research': 'mean',
            'leading_country': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A',
            'tech_detail': 'count'
        }).reset_index()

        # 컬럼명 변경
        category_data = category_data.rename(columns={'tech_detail': 'detail_count'})

        return df, category_data

    except Exception as e:
        st.error(f"데이터 로드 오류: {str(e)}")
        return None, None


# 경량화된 시각화 함수들
def create_simple_bar_comparison(data, title, metric_col, countries=['한국', '중국', '일본', '미국', 'EU']):
    """단순하고 빠른 막대그래프"""
    country_codes = ['kr', 'cn', 'jp', 'us', 'eu']
    values = [data[f'{code}_{metric_col}'].mean() for code in country_codes]

    fig = go.Figure(data=[
        go.Bar(
            x=countries,
            y=values,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
            text=[f"{val:.1f}%" if 'level' in metric_col else f"{val:.1f}년" for val in values],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title=title,
        height=300,
        yaxis=dict(range=[0, max(values) * 1.2])
    )

    return fig


def create_enhanced_heatmap(data, title="기술수준 히트맵"):
    """향상된 가시성의 히트맵"""
    countries = ['한국', '중국', '일본', '미국', 'EU']
    country_codes = ['kr', 'cn', 'jp', 'us', 'eu']

    # 상위 15개만 표시 (성능 최적화)
    top_data = data.nlargest(15, 'kr_tech_level') if len(data) > 15 else data

    heatmap_values = []
    heatmap_text = []

    for _, row in top_data.iterrows():
        values = [row[f'{code}_tech_level'] for code in country_codes]
        texts = [f"<b>{val:.1f}%</b>" for val in values]
        heatmap_values.append(values)
        heatmap_text.append(texts)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_values,
        x=countries,
        y=[name[:15] + "..." if len(name) > 15 else name for name in top_data['tech_category']],
        colorscale='RdYlGn',
        zmid=80,
        zmin=60,
        zmax=100,
        text=heatmap_text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},  # 폰트 크기 증대
        colorbar=dict(title=dict(text="기술수준(%)", font=dict(size=14)))
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),  # 제목 폰트 크기 증대
        height=max(400, len(top_data) * 40),
        xaxis=dict(title=dict(text="국가", font=dict(size=14))),
        yaxis=dict(title=dict(text="중분류", font=dict(size=14))),
        font=dict(size=12)
    )

    return fig


def create_radar_chart(data, selected_type='전체', selected_countries=['한국', '중국', '일본', '미국', 'EU']):
    """국가별 기술경쟁력 레이더 차트"""

    if selected_type != '전체':
        filtered_data = data[data['type'] == selected_type]
    else:
        filtered_data = data

    # 상위 8개 중분류만 표시 (성능 및 가독성)
    top_categories = filtered_data.nlargest(8, 'kr_tech_level')

    country_codes = {'한국': 'kr', '중국': 'cn', '일본': 'jp', '미국': 'us', 'EU': 'eu'}
    colors = {'한국': '#FF6B6B', '중국': '#4ECDC4', '일본': '#45B7D1', '미국': '#96CEB4', 'EU': '#FECA57'}

    # 레이더 차트용 데이터 생성
    radar_data = []
    for _, row in top_categories.iterrows():
        radar_entry = {
            'category': row['tech_category'][:10] + "..." if len(row['tech_category']) > 10 else row['tech_category']}
        for country in selected_countries:
            if country in country_codes:
                code = country_codes[country]
                radar_entry[country] = row[f'{code}_tech_level']
        radar_data.append(radar_entry)

    fig = go.Figure()

    for country in selected_countries:
        if country in country_codes:
            fig.add_trace(go.Scatterpolar(
                r=[entry[country] for entry in radar_data],
                theta=[entry['category'] for entry in radar_data],
                fill='toself',
                name=country,
                line_color=colors[country],
                fillcolor=colors[country],
                opacity=0.6
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        title=f"국가별 기술경쟁력 레이더 분석 ({selected_type})",
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

# 메인 애플리케이션
def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🌍 기후기술 수준조사 대시보드</h1>
        <p>NIGT 기후기술 수준조사 통계정보/분석</p>
    </div>
    """, unsafe_allow_html=True)

    # 데이터 로드
    with st.spinner('데이터를 로딩중입니다...'):
        df, category_data = load_climate_tech_data()

    if df is None or category_data is None:
        st.stop()

    # 사이드바
    st.sidebar.title("📊 분석 메뉴")

    analysis_type = st.sidebar.selectbox(
        "분석 유형을 선택하세요:",
        ["🏠 메인 대시보드", "🌏 국가별 경쟁력", "🔬 기술분야별 분석"]
    )

    # 메인 대시보드 - 2안(3패널 레이아웃)
    if analysis_type == "🏠 메인 대시보드":
        st.subheader("🇰🇷 메인 대시보드 - 한국 기후기술 경쟁력 현황")

        # 스토리
        st.markdown("""
        <div class="story-box">
            <h3>📖 한국 기후기술의 현재 위치</h3>
            <p>한국은 전체 기후기술에서 경쟁력 있는 분야와 개선이 필요한 분야가 공존합니다.
            본 패널은 한 화면에서 <strong>상세현황(중앙)</strong>, <strong>비교 그래프(왼쪽)</strong>, 
            <strong>핵심지표·히트맵·인사이트(오른쪽)</strong>를 동시에 제공합니다.</p>
        </div>
        """, unsafe_allow_html=True)

        # 범위 선택 (전체/감축/적응)
        scope = st.selectbox(
            "📊 분석 범위 선택:",
            ['전체', '감축기술', '적응기술'],
            key="scope_v2"
        )

        # 선택 데이터 필터
        if scope == '전체':
            filtered_data = category_data.copy()
            story_context = "전체 기후기술"
        elif scope == '감축기술':
            filtered_data = category_data[category_data['type'] == '감축']
            story_context = "감축기술"
        else:
            filtered_data = category_data[category_data['type'] == '적응']
            story_context = "적응기술"

        # 공통 지표 계산
        avg_kr_level = float(filtered_data['kr_tech_level'].mean())
        avg_kr_gap = float(filtered_data['kr_tech_gap'].mean())
        leading_count = int((filtered_data['kr_tech_group'] == '선도').sum())
        total_count = int(len(filtered_data))
        best_category = filtered_data.loc[filtered_data['kr_tech_level'].idxmax(), 'tech_category']

        # ===== 3 패널 레이아웃 =====
        left_col, center_col, right_col = st.columns([1, 2, 1], gap="large")

        # ---- 왼쪽 패널: 핵심지표 + 그래프 2개 ----
        with left_col:
            st.markdown("### 📊 한국 vs 주요국 기술수준 비교")
            st.caption(f"{story_context} 기준, 평균값 비교")
            fig_levels = create_simple_bar_comparison(filtered_data, "기술수준 비교(%)", "tech_level")
            st.plotly_chart(fig_levels, use_container_width=True, config={'displayModeBar': False})

            fig_gaps = create_simple_bar_comparison(filtered_data, "기술격차 비교(년)", "tech_gap")
            st.plotly_chart(fig_gaps, use_container_width=True, config={'displayModeBar': False})

        # ---- 중앙 패널(메인): 📋 상세현황 테이블 ----
        with center_col:

            st.markdown("### 🧭 핵심 지표")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("🇰🇷 평균 기술수준", f"{avg_kr_level:.1f}%",
                          delta="글로벌 3위" if avg_kr_level > 78 else "개선 필요")
                st.metric("🥇 선도 기술분야", f"{leading_count}개",
                          delta=f"전체 {total_count}개 중")
            with c2:
                st.metric("⏱️ 평균 기술격차", f"{avg_kr_gap:.1f}년",
                          delta="우수" if avg_kr_gap < 3 else "보통")
                st.metric("🏆 최우수 분야",
                          best_category[:12] + "..." if len(str(best_category)) > 12 else best_category)
                
            st.markdown("### 📋 전체 기후기술 상세현황")
            display_rows = []
            for _, row in filtered_data.iterrows():
                level_emoji = "🟢" if row['kr_tech_level'] >= 85 else "🟡" if row['kr_tech_level'] >= 70 else "🔴"
                gap_emoji = "🟢" if row['kr_tech_gap'] <= 2 else "🟡" if row['kr_tech_gap'] <= 4 else "🔴"
                group_emoji = {"선도": "🥇", "추격": "🥈", "후발": "🥉"}.get(row['kr_tech_group'], "❓")
                type_emoji = "⚡" if row['type'] == '감축' else "🛡️"

                display_rows.append({
                    '구분': f"{type_emoji} {row['type']}",
                    '중분류': row['tech_category'],
                    '한국 기술수준(%)': f"{level_emoji} {row['kr_tech_level']:.1f}%",
                    '한국 기술격차(년)': f"{gap_emoji} {row['kr_tech_gap']:.1f}년",
                    '한국 기술그룹': f"{group_emoji} {row['kr_tech_group']}",
                    '최고보유국': row['leading_country']
                })

            display_df = pd.DataFrame(display_rows)
            st.dataframe(
                display_df.sort_values('한국 기술수준(%)', ascending=False),
                use_container_width=True,
                hide_index=True,
                height=620
            )

        # ---- 오른쪽 패널: 히트맵 → 인사이트 ----
        with right_col:
            st.markdown("### 🔥 기술수준 히트맵 (상위 15)")
            fig_heatmap = create_enhanced_heatmap(filtered_data, f"{story_context} 기술수준 히트맵")
            st.plotly_chart(fig_heatmap, use_container_width=True, config={'displayModeBar': False})

            st.markdown("### 💡 핵심 인사이트")
            st.markdown(f"""
            <div class="insight-highlight">
                <p><strong>• 기술수준:</strong> 한국 {avg_kr_level:.1f}% (5개국 중 {'3위' if avg_kr_level > 78 else '4위'} 수준)</p>
                <p><strong>• 기술격차:</strong> 최고 수준 대비 평균 {avg_kr_gap:.1f}년 — {'우수' if avg_kr_gap < 3 else '보통' if avg_kr_gap < 4 else '개선 필요'}</p>
                <p><strong>• 경쟁 우위:</strong> 선도 {leading_count}개 분야, 최우수 분야는 <strong>{best_category}</strong></p>
            </div>
            """, unsafe_allow_html=True)

#------------------------------------------------------
    # 국가별 경쟁력 - 스토리보드 → (복원) 종합 비교 & 한국 상/하위 → 분석(3패널)
    elif analysis_type == "🌏 국가별 경쟁력":
        st.subheader("🌏 국가별 경쟁력")

        # ─── 스토리보드(유지) ───
        st.markdown("""
        <div class="story-box">
            <h3>📖 국가별 기후기술 경쟁력 비교분석 스토리보드</h3>
            <p>① 전체 통계를 확인합니다(종합 비교 · 한국 상/하위) → ② 상단 설정(국가/범위/세부축)을 정하고 → ③ 분석 3패널에서 심화 탐색합니다.</p>
        </div>
        """, unsafe_allow_html=True)

        # 공통 준비
        country_codes = {'한국': 'kr', '중국': 'cn', '일본': 'jp', '미국': 'us', 'EU': 'eu'}
        all_countries = list(country_codes.keys())

        # ─────────────────────────────────────────
        # (복원) 종합 비교 & 한국 상/하위 섹션 — 스토리보드 바로 아래
        # ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("📈 종합 비교 & 한국 상/하위 분야")

        wide_left, narrow_right = st.columns([2, 1], gap="large")

        # (좌) 종합 비교분석 - 전체 중분류 현황 (안정화 렌더링)
        with wide_left:
            st.markdown("#### 📊 종합 비교분석 - 전체 중분류 현황")

            # 1) 숫자 전용 DF 구성
            rows_num = []
            for _, r in category_data.iterrows():
                rows_num.append({
                    '구분': "⚡ 감축" if r['type'] == '감축' else "🛡️ 적응",
                    '중분류': r['tech_category'],
                    'KR': float(r.get('kr_tech_level', float('nan'))),
                    'CN': float(r.get('cn_tech_level', float('nan'))),
                    'JP': float(r.get('jp_tech_level', float('nan'))),
                    'US': float(r.get('us_tech_level', float('nan'))),
                    'EU': float(r.get('eu_tech_level', float('nan'))),
                    '최고보유국': r.get('leading_country', None)
                })
            num_df = pd.DataFrame(rows_num)

            # 2) 정렬 및 순위(한국 기준)
            num_df = num_df.sort_values('KR', ascending=False).reset_index(drop=True)
            num_df['순위'] = range(1, len(num_df) + 1)

            # 3) 표시용 DF (문자열)
            disp_df = num_df.copy()
            for col in ['KR', 'CN', 'JP', 'US', 'EU']:
                disp_df[col] = disp_df[col].apply(lambda v: f"{float(v):.1f}%" if v == v else "-")  # NaN 체크: v==v

            disp_df = disp_df.rename(columns={'KR': '한국', 'CN': '중국', 'JP': '일본', 'US': '미국', 'EU': 'EU'})
            disp_df = disp_df[['순위', '구분', '중분류', '한국', '중국', '일본', '미국', 'EU', '최고보유국']]
            disp_df = disp_df.fillna("-").astype(str)

            # 4) React #185 회피: HTML 테이블로 렌더
            def _html_table(df_):
                html = (
                    df_.to_html(index=False, escape=False)
                    .replace('<table border="1" class="dataframe">', '<table class="table" style="width:100%; border-collapse:collapse;">')
                    .replace('<th>', '<th style="text-align:center; padding:6px; border-bottom:1px solid #ddd;">')
                    .replace('<td>', '<td style="text-align:center; padding:6px; border-bottom:1px solid #f0f0f0;">')
                )
                return html

            st.markdown(_html_table(disp_df), unsafe_allow_html=True)

        # (우) 한국 상위/하위 10
        with narrow_right:
            st.markdown("#### 🏆 한국 상위 기술분야 (TOP 10)")
            top_categories = category_data.nlargest(10, 'kr_tech_level')
            top_table = []
            for _, row in top_categories.iterrows():
                top_table.append({
                    '구분': "⚡ 감축" if row['type'] == '감축' else "🛡️ 적응",
                    '중분류': row['tech_category'],
                    '기술수준(%)': f"{row['kr_tech_level']:.1f}%",
                    '기술격차(년)': f"{row['kr_tech_gap']:.1f}년"
                })
            st.dataframe(pd.DataFrame(top_table), hide_index=True, height=260)

            st.markdown("#### 📈 한국 개선 필요 분야 (하위 10)")
            bottom_categories = category_data.nsmallest(10, 'kr_tech_level')
            bottom_table = []
            for _, row in bottom_categories.iterrows():
                bottom_table.append({
                    '구분': "⚡ 감축" if row['type'] == '감축' else "🛡️ 적응",
                    '중분류': row['tech_category'],
                    '기술수준(%)': f"{row['kr_tech_level']:.1f}%",
                    '기술격차(년)': f"{row['kr_tech_gap']:.1f}년"
                })
            st.dataframe(pd.DataFrame(bottom_table), hide_index=True, height=260)

        # ─────────────────────────────────────────
        # 분석(3패널) 섹션 — 상단 컨트롤 + 3패널
        # ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("🧪 분석")

        # 상단 컨트롤: 분석 국가 / 분석 범위 / 레이더 축(세부기술)
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2], gap="large")

        with ctrl_col1:
            sel_country = st.selectbox("🌍 분석 국가", options=all_countries, index=0, key="prof_country_only")

        with ctrl_col2:
            scope = st.selectbox("📊 분석 범위", ['전체', '감축기술', '적응기술'], key="prof_scope_only")

        # 범위 필터 (중분류 테이블: category_data)
        if scope == '전체':
            scoped_cat = category_data.copy()
        elif scope == '감축기술':
            scoped_cat = category_data[category_data['type'] == '감축']
        else:
            scoped_cat = category_data[category_data['type'] == '적응']

        # 세부기술 후보(df 기반)
        if scope == '전체':
            detail_scope_df = df.copy()
        else:
            scoped_cats = set(scoped_cat['tech_category'].unique())
            detail_scope_df = df[df['tech_category'].isin(scoped_cats)].copy()

        with ctrl_col3:
            detail_opts = sorted(detail_scope_df['tech_detail'].dropna().unique().tolist())
            selected_details = st.multiselect("🎯 레이더 축(세부기술) 선택", options=detail_opts, default=[], key="prof_details_only")

        # ===== 3패널 =====
        left_col, center_col, right_col = st.columns([1, 2, 1], gap="large")

        # (좌) 핵심지표
        with left_col:
            st.markdown("### 🧭 핵심지표")
            code = country_codes[sel_country]
            avg_level = float(scoped_cat[f'{code}_tech_level'].mean()) if f'{code}_tech_level' in scoped_cat.columns else float('nan')
            avg_gap = float(scoped_cat[f'{code}_tech_gap'].mean()) if f'{code}_tech_gap' in scoped_cat.columns else float('nan')
            lead_cnt = int((scoped_cat['leading_country'] == sel_country).sum())
            total_cnt = int(len(scoped_cat))
            top_cat = "–"
            if f'{code}_tech_level' in scoped_cat.columns and not scoped_cat.empty:
                top_cat = str(scoped_cat.loc[scoped_cat[f'{code}_tech_level'].idxmax(), 'tech_category'])

            c1, c2 = st.columns(2)
            with c1:
                st.metric(f"{sel_country} 평균 기술수준", f"{avg_level:.1f}%")
                st.metric("선도 기술분야", f"{lead_cnt}개", delta=f"전체 {total_cnt}개 중")
            with c2:
                st.metric("평균 기술격차",
                          f"{avg_gap:.1f}년" if not pd.isna(avg_gap) else "데이터 없음",
                          delta="우수" if (not pd.isna(avg_gap) and avg_gap < 3) else "보통")
                st.metric("🏆 최우수 중분류", top_cat[:12] + "..." if len(top_cat) > 12 else top_cat)

        # (중앙) 선택국가 중분류 랭킹 (Top 12) — 세로막대
        with center_col:
            import plotly.express as px
            st.markdown("### 📊 중분류 랭킹 (Top 12)")
            if f'{code}_tech_level' in scoped_cat.columns:
                rank_df = (
                    scoped_cat[['tech_category', f'{code}_tech_level']]
                    .rename(columns={f'{code}_tech_level': 'level'})
                    .dropna(subset=['level'])
                    .sort_values('level', ascending=False)
                    .head(12)
                )
                fig_bar = px.bar(
                    rank_df,
                    x='tech_category', y='level',
                    labels={'level': '기술수준(%)', 'tech_category': '중분류'}
                )
                fig_bar.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_tickangle=-30)
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
            else:
                st.caption("※ 선택 국가의 기술수준 컬럼이 없어 랭킹 차트를 생성할 수 없습니다.")

        # (우) 레이더(세부기술) — 단일 국가 기준
        with right_col:
            import plotly.graph_objects as go
            st.markdown("### 🧭 레이더(세부기술)")
            # 자동 상위 8 축: 선택 없을 때, 선택국 기준
            if not selected_details:
                if f'{code}_tech_level' in detail_scope_df.columns:
                    auto_top = (
                        detail_scope_df
                        .dropna(subset=[f'{code}_tech_level'])
                        .nlargest(8, f'{code}_tech_level')
                    )
                else:
                    auto_top = detail_scope_df.head(8)
                radar_details = auto_top['tech_detail'].tolist()
                st.caption(f"※ 자동 축(상위 8, {sel_country} 기준): {', '.join(radar_details)}")
            else:
                radar_details = selected_details

            series_country, labels = [], []
            for tech in radar_details:
                row = detail_scope_df[detail_scope_df['tech_detail'] == tech]
                val = float(row.iloc[0][f'{code}_tech_level']) if (not row.empty and f'{code}_tech_level' in row.columns) else 0.0
                series_country.append(val)
                labels.append(tech[:12] + "..." if len(tech) > 12 else tech)

            fig_rad = go.Figure()
            fig_rad.add_trace(go.Scatterpolar(r=series_country, theta=labels, fill='toself',
                                              name=sel_country, opacity=0.6))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True, height=520,
                title=f"레이더(축: 세부기술) — {sel_country}"
            )
            st.plotly_chart(fig_rad, use_container_width=True, config={'displayModeBar': False})

#-----------------------------------------------------------------------------------------------------------------------
    # 기술분야별 분석 - 2안(3패널 레이아웃)
    elif analysis_type == "🔬 기술분야별 분석":
        st.subheader("🔬 기술분야별 상세 분석")

        # ----- 상단: 중분류 선택 -----
        col_sel1, col_sel2 = st.columns([3, 1])
        with col_sel1:
            selected_category = st.selectbox(
                "📋 중분류를 선택하세요:",
                options=sorted(category_data['tech_category'].unique()),
                key="category_select_v2"
            )
        with col_sel2:
            # 선택한 중분류에 대한 간단한 컨텍스트
            selected_scope = category_data.loc[category_data['tech_category'] == selected_category, 'type']
            scope_emoji = "⚡ 감축" if (len(selected_scope) > 0 and selected_scope.iloc[0] == '감축') else "🛡️ 적응"
            st.markdown(f"**구분:** {scope_emoji}")

        # ----- 선택 데이터 준비 -----
        # 중분류(카테고리) 단위 요약(=category_data의 한 행)
        cat_row_df = category_data[category_data['tech_category'] == selected_category].copy()
        if cat_row_df.empty:
            st.info("선택한 중분류에 해당하는 데이터가 없습니다.")
            st.stop()

        # 세부 기술(=df, 같은 중분류에 속한 하위 항목들)
        detail_df = df[df['tech_category'] == selected_category].copy()

        # 공통 지표 계산 (한국 기준)
        avg_kr_level = float(cat_row_df['kr_tech_level'].mean())
        avg_kr_gap = float(cat_row_df['kr_tech_gap'].mean())
        leading_count = int((cat_row_df['kr_tech_group'] == '선도').sum())
        total_count = int(len(cat_row_df))
        best_detail = None
        if not detail_df.empty and 'kr_tech_level' in detail_df.columns:
            best_detail = detail_df.loc[detail_df['kr_tech_level'].idxmax(), 'tech_detail']
        best_display = (best_detail[:12] + "...") if isinstance(best_detail, str) and len(best_detail) > 12 else (best_detail or selected_category)

        # ===== 3 패널 레이아웃 =====
        left_col, center_col, right_col = st.columns([1, 2, 1], gap="large")

        # ---- 왼쪽 패널: (1) 핵심지표 → (2) 한국 vs 주요국 비교 ----
        with left_col:
            st.markdown("### 📊 한국 vs 주요국 기술수준 비교")
            st.caption(f"중분류: {selected_category} 기준, 평균값 비교")
            # 기존 헬퍼 재사용: 단일 중분류(row) 전달해도 국가 막대 비교가 생성되도록 설계됨
            fig_levels = create_simple_bar_comparison(cat_row_df, "기술수준 비교(%)", "tech_level")
            st.plotly_chart(fig_levels, use_container_width=True, config={'displayModeBar': False})

            # (데이터가 있는 경우) 국가별 기술격차 비교
            try:
                fig_gaps = create_simple_bar_comparison(cat_row_df, "기술격차 비교(년)", "tech_gap")
                st.plotly_chart(fig_gaps, use_container_width=True, config={'displayModeBar': False})
            except Exception:
                st.caption("※ 국가별 기술격차 데이터 컬럼이 없는 경우 자동으로 생략됩니다.")

        # ---- 중앙 패널(메인): 📋 세부기술 상세현황 ----
        with center_col:
            st.markdown("### 🧭 핵심 지표")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("🇰🇷 평균 기술수준", f"{avg_kr_level:.1f}%",
                          delta="우수" if avg_kr_level >= 80 else "보통")
                st.metric("🥇 선도 기술분야", f"{leading_count}개",
                          delta=f"선택: {selected_category}")
            with c2:
                st.metric("⏱️ 평균 기술격차", f"{avg_kr_gap:.1f}년",
                          delta="우수" if avg_kr_gap < 3 else "보통")
                st.metric("🏆 최우수 세부기술", best_display)

            st.markdown("### 📋 세부기술 상세현황")
            if detail_df.empty:
                st.info("해당 중분류에 속한 세부기술 데이터가 없습니다.")
            else:
                rows = []
                for _, row in detail_df.iterrows():
                    level_emoji = "🟢" if row.get('kr_tech_level', 0) >= 85 else "🟡" if row.get('kr_tech_level', 0) >= 70 else "🔴"
                    gap_val = float(row.get('kr_tech_gap', 0)) if pd.notnull(row.get('kr_tech_gap', None)) else None
                    gap_emoji = "🟢" if (gap_val is not None and gap_val <= 2) else ("🟡" if (gap_val is not None and gap_val <= 4) else "🔴")
                    group_val = row.get('kr_tech_group', '–')
                    group_emoji = {"선도": "🥇", "추격": "🥈", "후발": "🥉"}.get(group_val, "❓")

                    rows.append({
                        '세부기술': row.get('tech_detail', '–'),
                        '한국 기술수준(%)': f"{level_emoji} {row.get('kr_tech_level', float('nan')):.1f}%",
                        '한국 기술격차(년)': f"{gap_emoji} {gap_val:.1f}년" if gap_val is not None else "–",
                        '한국 기술그룹': f"{group_emoji} {group_val}",
                        '최고보유국': row.get('leading_country', '–')
                    })
                display_df = pd.DataFrame(rows)
                st.dataframe(
                    display_df.sort_values('한국 기술수준(%)', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=620
                )

        # ---- 오른쪽 패널: 히트맵 → 인사이트 ----
        with right_col:
            st.markdown("### 🔥 기술수준 히트맵 (선택 중분류)")
            try:
                # 세부기술 단위 히트맵 (가능하면 detail_df 기반)
                fig_heatmap = create_enhanced_heatmap(detail_df if not detail_df.empty else cat_row_df,
                                                      f"{selected_category} 기술수준 히트맵")
                st.plotly_chart(fig_heatmap, use_container_width=True, config={'displayModeBar': False})
            except Exception:
                st.caption("※ 히트맵 생성에 필요한 컬럼이 부족하여 기본 형태로 대체되거나 생략될 수 있습니다.")

            st.markdown("### 💡 핵심 인사이트")
            st.markdown(f"""
            <div class="insight-highlight">
                <p><strong>• 기술수준:</strong> 한국 평균 {avg_kr_level:.1f}% — 해당 중분류 내 경쟁력 재점검 필요</p>
                <p><strong>• 기술격차:</strong> 평균 {avg_kr_gap:.1f}년 — {'우수' if avg_kr_gap < 3 else '보통' if avg_kr_gap < 4 else '개선 필요'}</p>
                <p><strong>• 세부 포커스:</strong> 최우수 세부기술은 <strong>{best_display}</strong></p>
            </div>
            """, unsafe_allow_html=True)

    # 사이드바 - 추가 정보
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 데이터 정보")
    st.sidebar.info(f"""
    **📈 데이터 현황**
    - 총 중분류: {len(category_data)}개
    - 총 세부기술: {len(df)}개  
    - 감축기술: {len(category_data[category_data['type'] == '감축'])}개 중분류
    - 적응기술: {len(category_data[category_data['type'] == '적응'])}개 중분류
    - 분석 국가: 5개국 (한국, 중국, 일본, 미국, EU)
    """)


if __name__ == "__main__":
    main()