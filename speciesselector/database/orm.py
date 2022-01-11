from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table, Column, Integer, ForeignKey, String, Enum, CheckConstraint, UniqueConstraint, Boolean, Float
from sqlalchemy.orm import relationship
import enum

# setup classes for data holding
Base = declarative_base()


class Path(Base):
    """stores paths (from user) on where to find data and more"""
    __tablename__ = 'path'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    value = Column(String)


class Species(Base):
    """stores each raw species info for a target dataset"""
    __tablename__ = 'species'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    split = Column(Integer)  #
    phylogenetic_weight = Column(Float)
    raw_results = relationship('RawResult', back_populates='test_species')


class RoundStatus(enum.Enum):
    initialized = 1  # round has been instantiated
    prepped = 2  # all symlinks and control files have been created
    seeds_training = 3
    adjustments_training = 4
    evaluated = 5


class Round(Base):
    """training rounds, planned [0, 3)"""
    __tablename__ = 'round'

    id = Column(Integer, primary_key=True)
    status = Column(Enum(RoundStatus))
    nni_seeds_id = Column(String)
    nni_adjustment_id = Column(String)
    nni_eval_id = Column(String)
    seed_models = relationship('SeedModel', back_populates='round')


class SeedTrainingSpecies(Base):
    """stores which species were in the seed training set in each round"""
    __tablename__ = 'seed_training_species'
    id = Column(Integer, primary_key=True)
    seed_model_id = Column(Integer, ForeignKey('seed_model.id'), nullable=False)
    seed_model = relationship('SeedModel', back_populates='seed_training_species')
    species_id = Column(Integer, ForeignKey('species.id'), nullable=False)
    species = relationship('Species')


class SeedModel(Base):
    """'best guess' training species at the start of each round"""
    __tablename__ = 'seed_model'
    id = Column(Integer, primary_key=True)
    round_id = Column(Integer, ForeignKey('round.id'), nullable=False)
    round = relationship('Round', back_populates='seed_models')
    split = Column(Integer)  # could also be derived from species below
    seed_training_species = relationship('SeedTrainingSpecies', back_populates='seed_model')

    @property
    def seed_validation_species(self):
        pass


class AdjustmentModel(Base):
    """stores adjustments fine-tuned from seed set and agglomerated results, each row is a (trained) model"""
    __tablename__ = 'adjustment_model'
    id = Column(Integer, primary_key=True)
    round_id = Column(Integer, ForeignKey('round.id'), nullable=False)
    species_id = Column(Integer, ForeignKey('species.id'))
    delta_n_species = Column(Integer)  # -1 for removed, 0 unchanged, 1 for added
    # raw_results will be agglomerated and weighted to fill weighted columns below
    raw_results = relationship('RawResult', back_populates='adjustment_model')
    weighted_test_genic_f1 = Column(Float)
    weighted_test_intergenic_f1 = Column(Float)
    weighted_test_utr_f1 = Column(Float)
    weighted_test_cds_f1 = Column(Float)
    weighted_test_intron_f1 = Column(Float)


class RawResult(Base):
    """each individual species evaluation for each adjusted model"""
    __tablename__ = 'raw_result'

    id = Column(Integer, primary_key=True)
    adjustment_model_id = Column(Integer, ForeignKey('adjustment_model.id'), nullable=False)
    adjustment_model = relationship('AdjustmentModel', back_populates='raw_results')
    test_species_id = Column(Integer, ForeignKey('species.id'))
    test_species = relationship('Species', back_populates='raw_results')
    genic_f1 = Column(Float)
    intergenic_f1 = Column(Float)
    utr_f1 = Column(Float)
    cds_f1 = Column(Float)
    intron_f1 = Column(Float)
