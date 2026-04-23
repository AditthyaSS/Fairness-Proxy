import type { Scenario } from '../types/api';

export interface FieldDef {
  value: any;
  type: 'merit' | 'protected';
}

export interface ScenarioPreset {
  label: string;
  domain: Scenario;
  targetEndpoint: string;
  liveDataset?: boolean;
  fields: Record<string, FieldDef>;
}

export const SCENARIOS: Record<string, ScenarioPreset> = {
  adult_income: {
    label: 'UCI Adult Income',
    domain: 'adult_income' as Scenario,
    targetEndpoint: '/v1/decisions/income',
    liveDataset: true,
    fields: {
      education: { value: 'Bachelors', type: 'merit' },
      'hours-per-week': { value: 40, type: 'merit' },
      workclass: { value: 'Private', type: 'merit' },
      occupation: { value: 'Prof-specialty', type: 'merit' },
      sex: { value: 'Female', type: 'protected' },
      race: { value: 'Black', type: 'protected' },
      age: { value: 35, type: 'protected' },
      'native-country': { value: 'United-States', type: 'protected' },
    },
  },
  loan_approval: {
    label: 'Loan Approval',
    domain: 'loan_approval' as Scenario,
    targetEndpoint: '/v1/decisions/loan',
    fields: {
      income: { value: 75000, type: 'merit' },
      credit_score: { value: 720, type: 'merit' },
      loan_amount: { value: 300000, type: 'merit' },
      employment_years: { value: 6, type: 'merit' },
      age: { value: 52, type: 'protected' },
      gender: { value: 'female', type: 'protected' },
      race: { value: 'Black', type: 'protected' },
      zip_code: { value: '10001', type: 'protected' },
    },
  },
  job_application: {
    label: 'Job Application',
    domain: 'job_application' as Scenario,
    targetEndpoint: '/v1/decisions/job',
    fields: {
      years_experience: { value: 4, type: 'merit' },
      education: { value: "Bachelor's", type: 'merit' },
      skills: { value: 'Python, SQL, ML', type: 'merit' },
      current_salary: { value: 65000, type: 'protected' },
      age: { value: 28, type: 'protected' },
      gender: { value: 'male', type: 'protected' },
      race: { value: 'Black', type: 'protected' },
    },
  },
  healthcare_triage: {
    label: 'Healthcare Triage',
    domain: 'healthcare_triage' as Scenario,
    targetEndpoint: '/v1/decisions/triage',
    fields: {
      symptoms: { value: 'chest pain', type: 'merit' },
      blood_pressure: { value: '140/90', type: 'merit' },
      heart_rate: { value: 88, type: 'merit' },
      age: { value: 67, type: 'protected' },
      race: { value: 'Hispanic', type: 'protected' },
      insurance: { value: 'Medicaid', type: 'protected' },
      income: { value: 28000, type: 'protected' },
    },
  },
  rental_application: {
    label: 'Rental Application',
    domain: 'rental_application' as Scenario,
    targetEndpoint: '/v1/decisions/rental',
    fields: {
      income: { value: 52000, type: 'merit' },
      credit_score: { value: 680, type: 'merit' },
      rental_history: { value: 'good', type: 'merit' },
      employment_status: { value: 'employed', type: 'merit' },
      age: { value: 34, type: 'protected' },
      race: { value: 'Black', type: 'protected' },
      familial_status: { value: 'two_children', type: 'protected' },
      disability: { value: true, type: 'protected' },
    },
  },
};

export const SCENARIO_ORDER: string[] = [
  'adult_income',
  'loan_approval',
  'job_application',
  'healthcare_triage',
  'rental_application',
];
