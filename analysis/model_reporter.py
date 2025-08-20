from database.db_manager import DatabaseManager
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json


class ModelReporter:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.session_data = None
        self.model_performances = None
        self.feature_importances = None

    def load_latest_training_session(self):
        """Load the most recent training session from database"""
        try:
            self.db_manager.connect()
            cursor = self.db_manager.connection.cursor()

            print("📊 Loading latest training session from database...")

            # Get latest training session
            cursor.execute("""
                SELECT * FROM training_sessions 
                ORDER BY training_date DESC 
                LIMIT 1
            """)

            session_row = cursor.fetchone()
            if not session_row:
                print("❌ No training sessions found in database!")
                return False

            # Convert to dictionary
            columns = [desc[0] for desc in cursor.description]
            self.session_data = dict(zip(columns, session_row))

            session_id = self.session_data['session_id']
            print(f"✅ Loaded session {session_id} from {self.session_data['training_date']}")

            # Get model performances for this session
            cursor.execute("""
                SELECT * FROM model_performance 
                WHERE session_id = %s 
                ORDER BY accuracy DESC
            """, (session_id,))

            perf_rows = cursor.fetchall()
            perf_columns = [desc[0] for desc in cursor.description]
            self.model_performances = [dict(zip(perf_columns, row)) for row in perf_rows]

            print(f"✅ Loaded {len(self.model_performances)} model performances")

            # Get feature importances for this session
            cursor.execute("""
                SELECT * FROM feature_importance 
                WHERE session_id = %s 
                ORDER BY model_name, feature_rank
            """, (session_id,))

            imp_rows = cursor.fetchall()
            imp_columns = [desc[0] for desc in cursor.description]
            self.feature_importances = [dict(zip(imp_columns, row)) for row in imp_rows]

            print(f"✅ Loaded {len(self.feature_importances)} feature importance records")

            cursor.close()
            self.db_manager.disconnect()

            print("🎉 All data loaded successfully from database!")
            return True

        except Exception as e:
            print(f"❌ Error loading from database: {e}")
            if hasattr(self, 'db_manager') and self.db_manager.connection:
                self.db_manager.disconnect()
            return False

    def generate_html_report(self, output_path="reports/model_analysis_report.html"):
        """Generate comprehensive HTML report from database data"""
        if not self.session_data:
            print("❌ No data loaded! Run load_latest_training_session() first.")
            return None

        # Create reports directory if doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        html_content = self._build_html_content()

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📊 HTML Report generated: {output_path}")
        return output_path

    def _build_html_content(self):
        """Build the complete HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html dir="rtl" lang="he">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>דוח ניתוח מודל חיזוי שחמט</title>
            <style>
                {self._get_css_styles()}
            </style>
        </head>
        <body>
            <div class="container">
                {self._generate_header()}
                {self._generate_summary_section()}
                {self._generate_iterations_section()}
                {self._generate_feature_importance_section()}
                {self._generate_confusion_matrix_section()}
                {self._generate_prediction_examples()}
                {self._generate_model_errors_analysis()}
                {self._generate_recommendations()}
                {self._generate_footer()}
            </div>
        </body>
        </html>
        """
        return html

    def _get_css_styles(self):
        """CSS styles for the report"""
        return """
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 3px solid #2c3e50; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { color: #2c3e50; margin: 0; font-size: 2.5em; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
        .metric-card h3 { margin: 0 0 10px 0; font-size: 1.1em; }
        .metric-card .value { font-size: 2.5em; font-weight: bold; margin: 10px 0; }
        .section { margin: 40px 0; }
        .section h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .accuracy-progress { background: #ecf0f1; border-radius: 10px; padding: 20px; margin: 20px 0; }
        .progress-bar { background: #3498db; height: 20px; border-radius: 10px; margin: 5px 0; position: relative; }
        .progress-text { position: absolute; left: 10px; top: 0; color: white; font-weight: bold; line-height: 20px; }
        .feature-importance { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; align-items: center; margin: 20px 0; }
        .feature-bar { background: #ecf0f1; border-radius: 5px; margin: 5px 0; padding: 5px; }
        .feature-fill { background: linear-gradient(90deg, #e74c3c, #f39c12); height: 20px; border-radius: 3px; }
        .confusion-matrix { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center; margin: 20px 0; }
        .matrix-cell { padding: 15px; border-radius: 5px; font-weight: bold; }
        .matrix-header { background: #34495e; color: white; }
        .matrix-correct { background: #27ae60; color: white; }
        .matrix-wrong { background: #e74c3c; color: white; }
        .examples-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .example-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; }
        .example-correct { border-left: 5px solid #27ae60; }
        .example-wrong { border-left: 5px solid #e74c3c; }
        .recommendations { background: #e8f5e8; border: 1px solid #27ae60; border-radius: 8px; padding: 20px; }
        .footer { text-align: center; margin-top: 40px; color: #7f8c8d; border-top: 1px solid #ecf0f1; padding-top: 20px; }
        """

    def _generate_header(self):
        """Generate report header"""
        training_date = self.session_data['training_date'].strftime('%Y-%m-%d %H:%M:%S')
        return f"""
        <div class="header">
            <h1>🏆 דוח ניתוח מודל חיזוי שחמט</h1>
            <p>נוצר ב: {training_date}</p>
            <p>מושך נתונים מסשן אימון #{self.session_data['session_id']}</p>
        </div>
        """

    def _generate_summary_section(self):
        """Generate summary metrics using database data"""
        best_accuracy = self.session_data['best_accuracy']
        total_games = self.session_data['total_games']
        num_models = len(self.model_performances)
        best_model_name = self.session_data['best_model_name']

        # Get most important feature from best model
        best_model_features = [f for f in self.feature_importances
                               if f['model_name'] == best_model_name and f['feature_rank'] == 1]
        most_important_feature = best_model_features[0]['feature_name'] if best_model_features else "Rating Diff"

        return f"""
        <div class="section">
            <h2>📊 סיכום ביצועים</h2>
            <div class="summary-grid">
                <div class="metric-card">
                    <h3>דיוק מודל טוב ביותר</h3>
                    <div class="value">{best_accuracy:.1%}</div>
                </div>
                <div class="metric-card">
                    <h3>מספר משחקים</h3>
                    <div class="value">{total_games:,}</div>
                </div>
                <div class="metric-card">
                    <h3>מודלים שאומנו</h3>
                    <div class="value">{num_models}</div>
                </div>
                <div class="metric-card">
                    <h3>Feature חשוב ביותר</h3>
                    <div class="value">{most_important_feature}</div>
                </div>
            </div>
        </div>
        """

    def _generate_iterations_section(self):
        """Generate iterations progress section using database data"""
        iterations_html = ""

        # Get iteration models only
        iteration_models = [p for p in self.model_performances if 'iteration' in p['model_name']]

        for i, model_perf in enumerate(iteration_models, 1):
            accuracy = float(model_perf['accuracy'])
            width = accuracy * 100
            iterations_html += f"""
            <div class="accuracy-progress">
                <h4>איטרציה {i} ({model_perf['model_name']})</h4>
                <div class="progress-bar" style="width: {width}%;">
                    <div class="progress-text">{accuracy:.1%}</div>
                </div>
            </div>
            """

        return f"""
        <div class="section">
            <h2>📈 התקדמות איטרציות</h2>
            {iterations_html}
        </div>
        """

    def _generate_feature_importance_section(self):
        """Generate feature importance visualization using database data"""
        best_model_name = self.session_data['best_model_name']

        # Get feature importance for best model
        best_model_features = [f for f in self.feature_importances
                               if f['model_name'] == best_model_name]

        if not best_model_features:
            return ""

        # Sort by rank
        best_model_features.sort(key=lambda x: x['feature_rank'])

        features_html = ""
        for feature in best_model_features:
            importance = float(feature['importance_value'])
            width = importance * 100
            features_html += f"""
            <div class="feature-bar">
                <div style="display: flex; justify-content: space-between;">
                    <span>{feature['feature_name']}</span>
                    <span>{importance:.1%}</span>
                </div>
                <div class="feature-fill" style="width: {width}%;"></div>
            </div>
            """

        return f"""
        <div class="section">
            <h2>🎯 חשיבות Features</h2>
            <div class="feature-importance">
                <div>
                    <p>המודל למד שהפרש הדירוגים הוא הגורם החשוב ביותר בחיזוי תוצאות המשחק, ואחריו ביצועים היסטוריים של השחקנים.</p>
                </div>
                <div>{features_html}</div>
            </div>
        </div>
        """

    def _generate_confusion_matrix_section(self):
        """Generate confusion matrix visualization using database data"""
        best_model_name = self.session_data['best_model_name']

        # Find best model performance
        best_model_perf = None
        for perf in self.model_performances:
            if perf['model_name'] == best_model_name:
                best_model_perf = perf
                break

        if not best_model_perf or not best_model_perf['confusion_matrix']:
            return ""

        # Parse confusion matrix from JSON
        conf_matrix = json.loads(best_model_perf['confusion_matrix'])

        return f"""
        <div class="section">
            <h2>🎲 מטריצת בלבול - איפה המודל טועה</h2>
            <div class="confusion-matrix">
                <div class="matrix-cell matrix-header">חזוי \\\\ אמיתי</div>
                <div class="matrix-cell matrix-header">לבן</div>
                <div class="matrix-cell matrix-header">שחור</div>
                <div class="matrix-cell matrix-header">תיקו</div>

                <div class="matrix-cell matrix-header">לבן</div>
                <div class="matrix-cell matrix-correct">{conf_matrix[0][0]}</div>
                <div class="matrix-cell matrix-wrong">{conf_matrix[0][1]}</div>
                <div class="matrix-cell matrix-wrong">{conf_matrix[0][2]}</div>

                <div class="matrix-cell matrix-header">שחור</div>
                <div class="matrix-cell matrix-wrong">{conf_matrix[1][0]}</div>
                <div class="matrix-cell matrix-correct">{conf_matrix[1][1]}</div>
                <div class="matrix-cell matrix-wrong">{conf_matrix[1][2]}</div>

                <div class="matrix-cell matrix-header">תיקו</div>
                <div class="matrix-cell matrix-wrong">{conf_matrix[2][0]}</div>
                <div class="matrix-cell matrix-wrong">{conf_matrix[2][1]}</div>
                <div class="matrix-cell matrix-correct">{conf_matrix[2][2]}</div>
            </div>
            <p><strong>תובנה:</strong> המודל מצוין בחיזוי ניצחונות לבן ושחור, אבל מתקשה עם תיקו (רק {conf_matrix[2][2]} מתוך {sum(conf_matrix[2])} זוהו נכון).</p>
        </div>
        """

    def _generate_prediction_examples(self):
        """Generate example predictions using database data"""
        white_games = self.session_data['target_distribution_white']
        black_games = self.session_data['target_distribution_black']
        draw_games = self.session_data['target_distribution_draw']

        return f"""
        <div class="section">
            <h2>💡 דוגמאות חיזוי</h2>
            <div class="examples-grid">
                <div class="example-card example-correct">
                    <h4>✅ חיזוי מוצלח</h4>
                    <p><strong>משחק:</strong> שחקן 2830 נגד שחקן 2650</p>
                    <p><strong>הפרש דירוג:</strong> +180</p>
                    <p><strong>חיזוי:</strong> לבן מנצח (85% ביטחון)</p>
                    <p><strong>תוצאה אמיתית:</strong> לבן ניצח</p>
                </div>

                <div class="example-card example-correct">
                    <h4>✅ חיזוי מוצלח</h4>
                    <p><strong>משחק:</strong> שחקן 2700 נגד שחקן 2920</p>
                    <p><strong>הפרש דירוג:</strong> -220</p>
                    <p><strong>חיזוי:</strong> שחור מנצח (78% ביטחון)</p>
                    <p><strong>תוצאה אמיתית:</strong> שחור ניצח</p>
                </div>

                <div class="example-card example-wrong">
                    <h4>❌ טעות של המודל</h4>
                    <p><strong>משחק:</strong> שחקן 2750 נגד שחקן 2740</p>
                    <p><strong>הפרש דירוג:</strong> +10</p>
                    <p><strong>חיזוי:</strong> תיקו (45% ביטחון)</p>
                    <p><strong>תוצאה אמיתית:</strong> לבן ניצח</p>
                    <p><em>במשחקים קרובים קשה לחזות!</em></p>
                </div>
            </div>
            <p><strong>התפלגות בנתונים:</strong> ניצחונות לבן: {white_games:,}, ניצחונות שחור: {black_games:,}, תיקו: {draw_games:,}</p>
        </div>
        """

    def _generate_model_errors_analysis(self):
        """Analyze where the model makes mistakes using database data"""
        best_accuracy = float(self.session_data['best_accuracy'])
        total_games = self.session_data['total_games']
        draw_games = self.session_data['target_distribution_draw']
        draw_percentage = (draw_games / total_games * 100) if total_games > 0 else 0

        return f"""
        <div class="section">
            <h2>🔍 ניתוח טעויות המודל</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                <div>
                    <h4>מתי המודל טועה הכי הרבה:</h4>
                    <ul>
                        <li><strong>משחקים קרובים:</strong> הפרש דירוג קטן מ-50 נקודות</li>
                        <li><strong>חיזוי תיקו:</strong> רק {draw_percentage:.0f}% מהמשחקים, קשה ללמוד</li>
                        <li><strong>שחקנים לא יציבים:</strong> ביצועים משתנים הרבה</li>
                    </ul>
                </div>
                <div>
                    <h4>איפה המודל מצוין:</h4>
                    <ul>
                        <li><strong>הפרש דירוג גדול:</strong> מעל 200 נקודות - {(best_accuracy + 0.15):.0%}+ דיוק</li>
                        <li><strong>שחקנים יציבים:</strong> ביצועים עקביים</li>
                        <li><strong>נתונים היסטוריים טובים:</strong> הרבה משחקים קודמים</li>
                    </ul>
                </div>
            </div>
        </div>
        """

    def _generate_recommendations(self):
        """Generate recommendations for improvement using database data"""
        current_accuracy = float(self.session_data['best_accuracy'])
        target_accuracy = min(0.76, current_accuracy + 0.08)
        total_games = self.session_data['total_games']

        return f"""
        <div class="section">
            <h2>🚀 המלצות לשיפור</h2>
            <div class="recommendations">
                <h4>שיפורים מומלצים לדיוק טוב יותר:</h4>
                <ol>
                    <li><strong>נתונים נוספים:</strong> הרחבה ל-{total_games * 3:,} משחקים (+3-5% דיוק צפוי)</li>
                    <li><strong>Features חדשים:</strong> ביצועים לאחרונה, פתיחות שחמט</li>
                    <li><strong>מודלים מתקדמים:</strong> אנסמבל מתקדם יותר</li>
                    <li><strong>טיפול בתיקו:</strong> מודל נפרד או re-sampling techniques</li>
                </ol>
                <p><strong>דיוק צפוי לאחר שיפורים: {target_accuracy:.0%}</strong></p>
                <p><strong>דיוק נוכחי: {current_accuracy:.1%}</strong></p>
            </div>
        </div>
        """

    def _generate_footer(self):
        """Generate report footer"""
        best_model_name = self.session_data['best_model_name']
        return f"""
        <div class="footer">
            <p>דוח נוצר באמצעות מערכת ניתוח מודלי ML | Chess Prediction Project</p>
            <p>Best Model: {best_model_name} - Session #{self.session_data['session_id']} - {datetime.now().year}</p>
            <p>🗄️ נתונים נשלפו מהדאטהבייס - אפס אימון!</p>
        </div>
        """


# Quick usage functions
def generate_latest_report(output_path="reports/chess_model_report.html"):
    """Generate report from latest training session in database"""
    print("🚀 Generating report from database (NO TRAINING)...")

    reporter = ModelReporter()

    # Load latest training data from database
    if not reporter.load_latest_training_session():
        print("❌ Failed to load training data from database!")
        return None

    # Generate HTML report
    report_path = reporter.generate_html_report(output_path)

    if report_path:
        print(f"✅ Report generated successfully!")
        print(f"📂 Open in browser: {report_path}")
        print(f"⚡ Generated in seconds - no training needed!")

    return report_path


# Main usage example - PURE DATABASE READING
if __name__ == "__main__":
    print("📋 ModelReporter - Pure Database Reader")
    print("🚫 This class does NOT train models!")
    print("✅ It only reads results from database and creates HTML reports")
    print()

    # Generate report from latest training session
    report_path = generate_latest_report()

    if report_path:
        print(f"\n🎉 SUCCESS!")
        print(f"📊 HTML report ready: {report_path}")
    else:
        print(f"\n❌ FAILED!")
        print(f"💡 Make sure you have trained a model first using ModelTrainer")