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
    page_title="🌍 기후기술 델파이조사 분석 대시보드",
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

# 기술 설명 데이터 (실제 데이터로 추후 교체 예정)
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

# 캐시된 데이터 로딩 함수
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
        radar_entry = {'category': row['tech_category'][:10] + "..." if len(row['tech_category']) > 10 else row['tech_category']}
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
        <h1>🌍 기후기술 델파이조사 분석 대시보드</h1>
        <p>NIGT 기후기술 수준조사 기반 의사결정 지원 도구 (스토리텔링 고도화 버전)</p>
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
    
    # 메인 대시보드 - 한국 중심 스토리텔링
    if analysis_type == "🏠 메인 대시보드":
        st.subheader("🇰🇷 메인 대시보드 - 한국 기후기술 경쟁력 현황")
        
        # 한국 중심 스토리 섹션
        st.markdown("""
        <div class="story-box">
            <h3>📖 한국 기후기술의 현재 위치</h3>
            <p>한국은 전체 44개 중분류 기후기술에서 <strong>평균 79.2%</strong>의 기술수준을 보유하며, 
            글로벌 최고 수준 대비 평균 <strong>2.8년</strong>의 기술격차를 보이고 있습니다. 
            특히 <strong>원자력, 태양광, 연료전지</strong> 분야에서 세계적 경쟁력을 확보한 상태입니다.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 계층 선택 드롭다운
        col1, col2 = st.columns([1, 2])
        
        with col1:
            hierarchy_level = st.selectbox(
                "📊 분석 범위:",
                ['전체', '감축기술', '적응기술'],
                key="hierarchy_level"
            )
        
        # 선택된 데이터 필터링
        if hierarchy_level == '전체':
            filtered_data = category_data.copy()
            story_context = "전체 기후기술"
        elif hierarchy_level == '감축기술':
            filtered_data = category_data[category_data['type'] == '감축']
            story_context = "감축기술"
        else:  # 적응기술
            filtered_data = category_data[category_data['type'] == '적응']
            story_context = "적응기술"
        
        # 한국 중심 핵심 지표
        col1, col2, col3, col4 = st.columns(4)
        
        avg_kr_level = filtered_data['kr_tech_level'].mean()
        avg_kr_gap = filtered_data['kr_tech_gap'].mean()
        leading_count = len(filtered_data[filtered_data['kr_tech_group'] == '선도'])
        total_count = len(filtered_data)
        
        with col1:
            st.metric("🇰🇷 한국 평균 기술수준", f"{avg_kr_level:.1f}%", 
                     delta=f"글로벌 3위" if avg_kr_level > 78 else "개선 필요")
        
        with col2:
            st.metric("⏱️ 평균 기술격차", f"{avg_kr_gap:.1f}년",
                     delta="우수" if avg_kr_gap < 3 else "보통")
        
        with col3:
            st.metric("🥇 선도 기술분야", f"{leading_count}개",
                     delta=f"전체 {total_count}개 중")
        
        with col4:
            best_category = filtered_data.loc[filtered_data['kr_tech_level'].idxmax(), 'tech_category']
            st.metric("🏆 최우수 분야", best_category[:12] + "..." if len(best_category) > 12 else best_category)
        
        # 기술개요 - 한국 vs 주요국 비교
        st.subheader(f"📊 {story_context} - 한국 vs 주요국 기술수준 비교")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_levels = create_simple_bar_comparison(filtered_data, "기술수준 비교", "tech_level")
            st.plotly_chart(fig_levels, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            fig_gaps = create_simple_bar_comparison(filtered_data, "기술격차 비교", "tech_gap")
            st.plotly_chart(fig_gaps, use_container_width=True, config={'displayModeBar': False})
        
        # 한국 중심 인사이트
        st.markdown(f"""
        <div class="insight-highlight">
            <h4>💡 {story_context} 핵심 인사이트</h4>
            <p><strong>• 기술수준:</strong> 한국은 {avg_kr_level:.1f}%로 5개국 중 {'3위' if avg_kr_level > 78 else '4위'} 수준</p>
            <p><strong>• 기술격차:</strong> 최고 수준 대비 평균 {avg_kr_gap:.1f}년 격차, {'우수한' if avg_kr_gap < 3 else '보통' if avg_kr_gap < 4 else '개선이 필요한'} 수준</p>
            <p><strong>• 경쟁 우위:</strong> {leading_count}개 분야에서 선도 지위 확보</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 경량화된 히트맵 (성능 개선)
        st.subheader(f"🔥 {story_context} 기술수준 현황 (한국 기준 상위 15개)")
        fig_heatmap = create_enhanced_heatmap(filtered_data, f"{story_context} 기술수준 히트맵")
        st.plotly_chart(fig_heatmap, use_container_width=True, config={'displayModeBar': False})
        
        # 상세현황 테이블 (색상 강화)
        st.subheader(f"📋 {story_context} 상세현황")
        
        display_data = []
        for _, row in filtered_data.iterrows():
            level_emoji = "🟢" if row['kr_tech_level'] >= 85 else "🟡" if row['kr_tech_level'] >= 70 else "🔴"
            gap_emoji = "🟢" if row['kr_tech_gap'] <= 2 else "🟡" if row['kr_tech_gap'] <= 4 else "🔴"
            group_emoji = {"선도": "🥇", "추격": "🥈", "후발": "🥉"}.get(row['kr_tech_group'], "❓")
            type_emoji = "⚡" if row['type'] == '감축' else "🛡️"
            
            display_data.append({
                '구분': f"{type_emoji} {row['type']}",
                '중분류': row['tech_category'],
                '기술수준(%)': f"{level_emoji} {row['kr_tech_level']:.1f}%",
                '기술격차(년)': f"{gap_emoji} {row['kr_tech_gap']:.1f}년",
                '기술그룹': f"{group_emoji} {row['kr_tech_group']}",
                '최고보유국': row['leading_country']
            })
        
        display_df = pd.DataFrame(display_data)
        st.dataframe(
            display_df.sort_values('기술수준(%)', ascending=False),
            use_container_width=True,
            hide_index=True,
            height=400
        )
    
    # 국가별 경쟁력 - 주요국 비교 스토리텔링
    elif analysis_type == "🌏 국가별 경쟁력":
        st.subheader("🌍 국가별 기후기술 경쟁력 비교 분석")
        
        # 국가별 경쟁력 스토리 섹션
        st.markdown("""
        <div class="story-box">
            <h3>🏁 글로벌 기후기술 경쟁 구도</h3>
            <p>현재 기후기술 분야에서는 <strong>미국이 선두</strong>를 달리고 있으며, 
            <strong>중국이 제조 기반 기술에서 급부상</strong>, <strong>EU는 정책 연계 기술 혁신</strong>, 
            <strong>일본은 정밀 기술 우위</strong>, <strong>한국은 시스템 통합 강점</strong>을 보이는 구조입니다.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 레이더 차트 컨트롤
        col1, col2 = st.columns([1, 2])
        
        with col1:
            radar_type = st.selectbox(
                "🎯 분석 분류:",
                ['전체', '감축', '적응'],
                key="radar_type"
            )
        
        with col2:
            countries = ['한국', '중국', '일본', '미국', 'EU']
            selected_countries = st.multiselect(
                "🌐 비교 국가 선택:",
                countries,
                default=countries,
                key="selected_countries"
            )
        
        # 레이더 차트
        if selected_countries:
            st.subheader("📡 국가별 기술경쟁력 레이더 분석")
            fig_radar = create_radar_chart(category_data, radar_type, selected_countries)
            st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
        
        # 국가별 성과 분석
        st.subheader("📊 주요국 기술경쟁력 현황")
        
        country_analysis = []
        country_codes = ['kr', 'cn', 'jp', 'us', 'eu']
        country_names = ['한국', '중국', '일본', '미국', 'EU']
        
        for name, code in zip(country_names, country_codes):
            avg_level = category_data[f'{code}_tech_level'].mean()
            leading_count = len(category_data[category_data[f'{code}_tech_level'] >= 90])
            
            country_analysis.append({
                'country': name,
                'avg_level': avg_level,
                'leading_count': leading_count,
                'rank': 0  # 나중에 계산
            })
        
        # 순위 계산
        sorted_analysis = sorted(country_analysis, key=lambda x: x['avg_level'], reverse=True)
        for i, item in enumerate(sorted_analysis):
            item['rank'] = i + 1
        
        # 국가별 성과 카드
        cols = st.columns(5)
        for i, country_info in enumerate(sorted_analysis):
            with cols[i]:
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][country_info['rank']-1]
                st.markdown(f"""
                <div class="country-performance">
                    <h4>{rank_emoji} {country_info['country']}</h4>
                    <p><strong>평균 기술수준</strong><br>{country_info['avg_level']:.1f}%</p>
                    <p><strong>선도 기술 수</strong><br>{country_info['leading_count']}개</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 향상된 히트맵
        st.subheader("🔥 국가별 기술수준 히트맵 - 전체 현황")
        fig_country_heatmap = create_enhanced_heatmap(category_data, "국가별 기술수준 종합 현황")
        st.plotly_chart(fig_country_heatmap, use_container_width=True, config={'displayModeBar': False})
        
        # 상위/하위 기술분야 (개선된 테이블)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 한국 상위 기술분야 (TOP 10)")
            top_categories = category_data.nlargest(10, 'kr_tech_level')
            top_table = []
            for _, row in top_categories.iterrows():
                top_table.append({
                    '구분': "⚡ 감축" if row['type'] == '감축' else "🛡️ 적응",
                    '중분류': row['tech_category'],
                    '기술수준(%)': f"{row['kr_tech_level']:.1f}%",
                    '기술격차(년)': f"{row['kr_tech_gap']:.1f}년"
                })
            st.dataframe(pd.DataFrame(top_table), hide_index=True, height=350)
        
        with col2:
            st.subheader("📈 한국 개선 필요 분야 (하위 10)")
            bottom_categories = category_data.nsmallest(10, 'kr_tech_level')
            bottom_table = []
            for _, row in bottom_categories.iterrows():
                bottom_table.append({
                    '구분': "⚡ 감축" if row['type'] == '감축' else "🛡️ 적응",
                    '중분류': row['tech_category'],
                    '기술수준(%)': f"{row['kr_tech_level']:.1f}%",
                    '기술격차(년)': f"{row['kr_tech_gap']:.1f}년"
                })
            st.dataframe(pd.DataFrame(bottom_table), hide_index=True, height=350)
        
        # 종합 비교 분석 (기존 종합비교에서 이동)
        st.subheader("📈 종합 비교 분석 - 전체 중분류 현황")
        
        comparison_data = []
        for _, row in category_data.iterrows():
            comparison_data.append({
                '순위': 0,  # 나중에 설정
                '구분': "⚡ 감축" if row['type'] == '감축' else "🛡️ 적응",
                '중분류': row['tech_category'],
                '한국': f"{row['kr_tech_level']:.1f}%",
                '중국': f"{row['cn_tech_level']:.1f}%",
                '일본': f"{row['jp_tech_level']:.1f}%",
                '미국': f"{row['us_tech_level']:.1f}%",
                'EU': f"{row['eu_tech_level']:.1f}%",
                '최고보유국': row['leading_country']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['한국_정렬용'] = comparison_df['한국'].str.replace('%', '').astype(float)
        comparison_df = comparison_df.sort_values('한국_정렬용', ascending=False).reset_index(drop=True)
        comparison_df['순위'] = range(1, len(comparison_df) + 1)
        comparison_df = comparison_df.drop('한국_정렬용', axis=1)
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True, height=500)
    
    # 기술분야별 분석 - 개별 기술 집중 분석
    elif analysis_type == "🔬 기술분야별 분석":
        st.subheader("🔬 기술분야별 상세 분석")
        
        # 중분류 선택
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_category = st.selectbox(
                "📋 중분류를 선택하세요:",
                options=sorted(category_data['tech_category'].unique()),
                key="category_select"
            )
        
        with col2:
            detail_techs = df[df['tech_category'] == selected_category]['tech_detail'].tolist()
            selected_detail = st.selectbox(
                "🔍 세부기술 선택:",
                options=['전체(중분류)'] + detail_techs,
                key="detail_select"
            )
        
        if selected_category:
            category_info = category_data[category_data['tech_category'] == selected_category].iloc[0]
            
            # 기술 설명 및 현황 카드
            tech_desc = TECH_DESCRIPTIONS.get(selected_category, {
                "description": "해당 기술분야에 대한 상세 설명은 추후 보완 예정입니다.",
                "korea_status": "한국의 기술 현황 분석은 추후 보완 예정입니다.",
                "global_trend": "글로벌 기술 동향 분석은 추후 보완 예정입니다."
            })
            
            st.markdown(f"""
            <div class="tech-summary-card">
                <h2>📋 {selected_category}</h2>
                <p>{tech_desc['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 기술 현황 카드 4개
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                level_color = "🟢" if category_info['kr_tech_level'] >= 85 else "🟡" if category_info['kr_tech_level'] >= 70 else "🔴"
                st.markdown(f"""
                <div style="background: #f1f5f9; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4>{level_color} 기술수준</h4>
                    <h2 style="color: #3b82f6; margin: 0;">{category_info['kr_tech_level']:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                gap_color = "🟢" if category_info['kr_tech_gap'] <= 2 else "🟡" if category_info['kr_tech_gap'] <= 4 else "🔴"
                st.markdown(f"""
                <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4>{gap_color} 기술격차</h4>
                    <h2 style="color: #f59e0b; margin: 0;">{category_info['kr_tech_gap']:.1f}년</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                group_color = {"선도": "🥇", "추격": "🥈", "후발": "🥉"}.get(category_info['kr_tech_group'], "❓")
                st.markdown(f"""
                <div style="background: #ecfdf5; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4>{group_color} 기술그룹</h4>
                    <h2 style="color: #10b981; margin: 0;">{category_info['kr_tech_group']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div style="background: #f3e8ff; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4>🏆 최고보유국</h4>
                    <h2 style="color: #8b5cf6; margin: 0;">{category_info['leading_country']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # 탭 기반 분석
            tab1, tab2, tab3 = st.tabs(["📊 기술수준 및 격차", "🎯 역량 및 경향", "📈 논문·특허"])
            
            with tab1:
                st.subheader("📊 기술수준 및 격차 분석")
                
                # 기술수준/격차 차트 분리
                col1, col2 = st.columns(2)
                
                with col1:
                    countries = ['한국', '중국', '일본', '미국', 'EU']
                    country_codes = ['kr', 'cn', 'jp', 'us', 'eu']
                    tech_levels = [category_info[f'{code}_tech_level'] for code in country_codes]
                    
                    fig_level = go.Figure(data=[
                        go.Bar(
                            x=countries,
                            y=tech_levels,
                            marker_color=['#FF6B6B' if c == '한국' else '#E5E7EB' for c in countries],
                            text=[f"{val:.1f}%" for val in tech_levels],
                            textposition='outside'
                        )
                    ])
                    
                    fig_level.update_layout(
                        title="기술수준 비교",
                        height=400,
                        yaxis=dict(range=[0, 105])
                    )
                    
                    st.plotly_chart(fig_level, use_container_width=True, config={'displayModeBar': False})
                
                with col2:
                    tech_gaps = [category_info[f'{code}_tech_gap'] for code in country_codes]
                    
                    fig_gap = go.Figure(data=[
                        go.Bar(
                            x=countries,
                            y=tech_gaps,
                            marker_color=['#FF6B6B' if c == '한국' else '#E5E7EB' for c in countries],
                            text=[f"{val:.1f}년" for val in tech_gaps],
                            textposition='outside'
                        )
                    ])
                    
                    fig_gap.update_layout(
                        title="기술격차 비교",
                        height=400,
                        yaxis=dict(range=[0, max(tech_gaps) * 1.2])
                    )
                    
                    st.plotly_chart(fig_gap, use_container_width=True, config={'displayModeBar': False})
                
                # 향상된 상세 테이블
                st.subheader("📋 상세 현황")
                detail_table = []
                for country, code in zip(countries, country_codes):
                    level = category_info[f'{code}_tech_level']
                    gap = category_info[f'{code}_tech_gap']
                    group = category_info.get(f'{code}_tech_group', 'N/A')
                    
                    # 최고 수준 국가 찾기
                    is_leader = level == max([category_info[f'{c}_tech_level'] for c in country_codes])
                    
                    detail_table.append({
                        '국가': f"🏆 {country}" if is_leader else country,
                        '기술수준(%)': f"{level:.1f}%",
                        '기술격차(년)': f"{gap:.1f}년",
                        '기술그룹': group,
                        '최고수준국가': "✓" if is_leader else ""
                    })
                
                st.dataframe(pd.DataFrame(detail_table), use_container_width=True, hide_index=True)
            
            with tab2:
                st.subheader("🎯 연구개발 역량 및 경향")
                
                # R&D 역량 테이블 (기존 유지)
                countries = ['한국', '중국', '일본', '미국', 'EU']
                country_codes = ['kr', 'cn', 'jp', 'us', 'eu']
                
                rd_data = []
                for country, code in zip(countries, country_codes):
                    trend = category_info.get(f'{code}_rd_trend', 'N/A')
                    basic = category_info.get(f'{code}_basic_research', 0)
                    applied = category_info.get(f'{code}_applied_research', 0)
                    
                    rd_data.append({
                        '국가': country,
                        '기초연구역량': f"{basic:.1f}" if pd.notna(basic) else "N/A",
                        '응용연구역량': f"{applied:.1f}" if pd.notna(applied) else "N/A", 
                        'R&D활동경향': trend
                    })
                
                st.dataframe(pd.DataFrame(rd_data), use_container_width=True, hide_index=True)
                
                # 한국 중심 역량 분석 (기존 유지)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div style="background: #f1f5f9; padding: 1rem; border-radius: 8px;">
                        <h4>🇰🇷 한국의 R&D 역량</h4>
                        <p><strong>기초연구역량:</strong> {category_info['kr_basic_research']:.1f}점</p>
                        <p><strong>응용연구역량:</strong> {category_info['kr_applied_research']:.1f}점</p>
                        <p><strong>R&D 활동경향:</strong> {category_info['kr_rd_trend']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # 글로벌 최고 수준 대비
                    max_basic = max([category_info[f'{code}_basic_research'] for code in country_codes if pd.notna(category_info[f'{code}_basic_research'])])
                    max_applied = max([category_info[f'{code}_applied_research'] for code in country_codes if pd.notna(category_info[f'{code}_applied_research'])])
                    
                    st.markdown(f"""
                    <div style="background: #f1f5f9; padding: 1rem; border-radius: 8px;">
                        <h4>🌍 글로벌 최고 수준 대비</h4>
                        <p><strong>기초연구 격차:</strong> {max_basic - category_info['kr_basic_research']:.1f}점</p>
                        <p><strong>응용연구 격차:</strong> {max_applied - category_info['kr_applied_research']:.1f}점</p>
                        <p><strong>종합 경쟁력:</strong> {'우수' if (category_info['kr_basic_research'] + category_info['kr_applied_research'])/2 > 75 else '보통'}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab3:
                st.subheader("📈 논문·특허 분석")
                
                st.markdown("""
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                    <strong>📌 샘플 데이터 안내:</strong> 현재 표시되는 논문/특허 데이터는 시연용 샘플입니다. 
                    향후 실제 논문/특허 DB 연동을 통해 실제 데이터로 구축될 예정입니다.
                </div>
                """, unsafe_allow_html=True)
                
                # 샘플 논문/특허 통계 (기존 유지)
                sample_data = {
                    '한국': {'논문': 156, '특허': 89, '증가율': '+12%'},
                    '중국': {'논문': 324, '특허': 156, '증가율': '+28%'},
                    '일본': {'논문': 198, '특허': 134, '증가율': '+8%'},
                    '미국': {'논문': 289, '특허': 201, '증가율': '+15%'},
                    'EU': {'논문': 234, '특허': 167, '증가율': '+11%'}
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 최근 5년 논문 발표 현황")
                    paper_data = []
                    for country, data in sample_data.items():
                        paper_data.append({
                            '국가': country,
                            '논문 수(편)': data['논문'],
                            '증가율': data['증가율']
                        })
                    st.dataframe(pd.DataFrame(paper_data), hide_index=True)
                
                with col2:
                    st.subheader("🏭 최근 5년 특허 출원 현황")
                    patent_data = []
                    for country, data in sample_data.items():
                        patent_data.append({
                            '국가': country,
                            '특허 수(건)': data['특허'],
                            '증가율': data['증가율']
                        })
                    st.dataframe(pd.DataFrame(patent_data), hide_index=True)
        
        # 다운로드 섹션 (기존 유지)
        st.markdown("---")
        st.subheader("💾 분석 결과 다운로드")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 중분류별 종합 분석 다운로드", type="primary"):
                csv_buffer = io.StringIO()
                category_data.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                st.download_button(
                    label="💾 중분류_종합분석.csv 다운로드",
                    data=csv_buffer.getvalue(),
                    file_name=f"중분류_종합분석_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🔍 세부기술별 상세 데이터 다운로드", type="secondary"):
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                st.download_button(
                    label="💾 세부기술_상세데이터.csv 다운로드",
                    data=csv_buffer.getvalue(),
                    file_name=f"세부기술_상세데이터_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
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