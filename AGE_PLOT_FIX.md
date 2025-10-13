# Age Group Plot Fix ✅

## 🐛 Issue Identified

The **Age Group plot** in Act III Slide 3.2 was showing:
> "Unable to load age group data. Please ensure Module 2 data is loaded or database connection is available."

## 🔍 Root Cause

The `unemployed_by_age_sex_long` table **doesn't have an 'occupation' column** in some cases. The original `_load_age_data()` function was:
1. Looking for an 'occupation' column directly
2. Returning `None` if not found
3. Not handling the flexible column naming used in the actual database tables

## ✅ Solution Applied

Updated `_load_age_data()` to **match Module 3's approach**:

### **1. Added Helper Function**
```python
def _find_column(df: pd.DataFrame, keywords: list[str]):
    """Find column by keywords (case-insensitive)"""
```
- Searches for columns using keyword matching (like Module 3 does)
- Case-insensitive search
- Returns first match or None

### **2. Updated Age Data Loading**
- Uses `_find_column()` to flexibly find:
  - Age column: `['age_group', 'ageband', 'age bracket', 'age']`
  - Occupation column: `['occupation']`
  - Count column: `['unemployed_count', 'unemployment_count', 'unemp_count', 'count']`

- **Handles missing occupation column:**
  - If no occupation column found → Creates `'Overall'` category
  - Matches Module 3's `prepare_demographic_share()` behavior

- **Collapses gender dimension:**
  - Groups by occupation + age group
  - Sums counts across gender categories

- **Selects top occupations:**
  - If only one occupation (e.g., 'Overall') → Uses all data
  - Otherwise → Selects top 4 by total count

## 🔧 Technical Changes

**File:** `app_pages/presentation_slides.py`

**Added:**
- `_find_column()` helper function (~10 lines)

**Modified:**
- `_load_age_data()` function (~50 lines)
  - Flexible column detection
  - Handles missing occupation column
  - Gender collapse logic
  - Proper error handling

## 📊 Expected Behavior Now

### **If age table has occupations:**
- Shows top 4 occupations
- Age group distribution within each occupation
- Stacked bars totaling 100% per occupation

### **If age table doesn't have occupations:**
- Shows 'Overall' category
- Age group distribution across all unemployed
- Single stacked bar showing age breakdown

### **Both cases:**
- Collapses gender for cleaner visualization
- Calculates share percentages correctly
- Displays real data from database

## 🧪 Testing

1. **Navigate to Module 2** and load unemployment dataset
2. **Enter Presentation Mode** → **Act III** → **Slide 3.2**
3. **Click "👥 Age Groups" button**
4. **Verify:**
   - [ ] Modal opens successfully
   - [ ] Stacked bar chart displays
   - [ ] Age groups shown (15-24, 25-34, etc.)
   - [ ] Either multiple occupations OR 'Overall' category
   - [ ] Bars stack to 100%
   - [ ] No error messages

## 💡 Why This Fix Works

The original code assumed the age table would have an 'occupation' column with the exact name "occupation". In reality:

- **Column names vary** (case differences, variations like "occupation_name")
- **Some age tables are overall statistics** (no occupation breakdown)
- **Module 3 handles this gracefully** with flexible column detection

By adopting Module 3's approach, we now:
✅ **Handle column name variations**
✅ **Work with or without occupation breakdowns**
✅ **Match Module 3's data processing exactly**
✅ **Provide consistent visualization regardless of data structure**

## 🚀 Status

**Issue:** Age group plot not loading ❌  
**Fix Applied:** Flexible column detection + missing occupation handling ✅  
**Testing:** Ready for user verification 🧪  
**Expected Result:** Plot displays age distributions from real data 📊

---

The age group plot should now work correctly! Try clicking the "👥 Age Groups" button in Slide 3.2 to verify the fix.
